import io
import cv2
import torch
import gdown
import base64
import librosa
import tempfile
import soundfile
import numpy as np
import pickle as pl
from math import ceil
from tqdm import tqdm
import streamlit as st
from pathlib import Path
from copy import deepcopy
import scipy.signal as sps
from functools import partial
from pydub import AudioSegment
from consolidate import time_decoder
from split_texts import split_by_words
from make_xml import make_xml_from_words
from transformers import Wav2Vec2Processor
from moviepy.editor import VideoFileClip, TextClip
from moviepy.video.tools.subtitles import SubtitlesClip
from flash.core.data.data_source import DefaultDataKeys
from flash.audio import SpeechRecognition, SpeechRecognitionData


# stop multiprocessing
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@st.experimental_singleton
def model_and_processor():
    # load model and processor once
    backbone = "facebook/wav2vec2-base-960h"
    model = SpeechRecognition(backbone)
    processor = Wav2Vec2Processor.from_pretrained(backbone)
    return model, processor


model, processor = model_and_processor()


def pickle(data, filename="data.pickle"):
    with open(filename, "wb") as file_handler:
        pl.dump(data, file_handler)


def unpickle(filename="data.pickle"):
    with open(filename, "rb") as file_handler:
        return pl.load(file_handler)


def download_data(data):
    output_data = pl.dumps(data)
    b64 = base64.b64encode(output_data).decode()
    href = f'<a href="data:file/output_data;base64,{b64}" download="myfile.pkl">Download data .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)


def resample_numpy(audio_numpy, sample_rate):
    if sample_rate == 16000:
        return audio_numpy

    new_rate = 16000
    number_of_samples = round(len(audio_numpy) * float(new_rate) / sample_rate)
    resampled_audio = sps.resample(audio_numpy, number_of_samples)
    return resampled_audio


def convert_audio_file(audio_bytes) -> None:
    try:
        clip, sample_rate = soundfile.read(audio_bytes)
        # make sure the audio is the right shape
        try:
            if clip.shape[0] > clip.shape[1]:
                clip = clip.transpose()
        except IndexError:
            pass
        # convert to mono
        clip = librosa.to_mono(clip)
    
    except RuntimeError:
        audio_bytes.seek(0)
        audio_object = AudioSegment.from_file_using_temporary_files(audio_bytes)
        
        # convert to mono
        if audio_object.channels == 1:
            clip = np.array(audio_object.get_array_of_samples())
        elif audio_object.channels > 1:
            # convert to mono
            channels = audio_object.split_to_mono()
            channel_list = [np.array(i.get_array_of_samples()) for i in channels]
            clip = np.mean(channel_list, axis=0)
        
        sample_rate = audio_object.frame_rate
    resampled_audio = resample_numpy(clip, sample_rate)
    return resampled_audio


def transcribe_audio(audio_numpy):
    try:
        if audio_numpy.shape[0] > audio_numpy.shape[1]:
            audio_numpy = audio_numpy.transpose()
    except IndexError:
        pass
    # convert to mono
    audio_numpy = librosa.to_mono(audio_numpy)

    input_ = processor(audio_numpy)
    input_ = input_["input_values"][0]

    audio_length = len(input_)
    # split = 148821
    split = 78821

    audio_split = [i * split for i in range(ceil(audio_length / split))]

    audio_list = np.split(input_, np.array(audio_split))

    decoded = []
    batch_decoded = []

    for audio in tqdm(audio_list):
        if len(audio) == 0:
            continue

        audio_dict = {
            DefaultDataKeys.INPUT: audio,
            DefaultDataKeys.TARGET: "dummy target",
            DefaultDataKeys.METADATA: {"sampling_rate": 16000},
        }

        # predictions = model.predict([custom_datamodule._test_ds[0]])

        predictions = model.predict([audio_dict])

        pred_ids = torch.argmax(predictions[0].logits, dim=-1)

        decoded.extend(processor.decode(pred_ids))
        decoded.extend([" "])  # add space between this decoding and the next
        batch_decoded.extend(processor.batch_decode(pred_ids))

    decoded = "".join(decoded)  # combine into one string

    return decoded, batch_decoded


def split_word_list(decoded, word_start, word_end):
    word_list = decoded.split()

    word_list, word_start, word_end = split_by_words(
        word_list=word_list, word_start=word_start, word_end=word_end
    )

    return word_list, word_start, word_end


def parse_text_box():
    # use fuzzy matching to match rows?
    # use Git to compare lines?
    pass


def word_generator(word_list):
    # skip first frame
    yield ""
    for word in word_list:
        yield word
    # return nothing for the rest of the frames
    while True:
        yield ""


def text_clip(text: str):
    """
    Return a description string on the bottom-left of the video

    Args:
                text (str): Text to show

    Returns:
                moviepy.editor.TextClip: A instance of a TextClip
    """
    my_text = TextClip(text, font="Helvetica", fontsize=50, color="white").set_position(
        ("center", "center")
    )

    return my_text.on_color(
        size=(my_text.w, my_text.h),
        color=(0, 0, 0),
        pos=("center", "center"),
        col_opacity=0.6,
    )


def process_audio(audio_numpy):
    decoded, batch_decoded = transcribe_audio(audio_numpy)
    word_start, word_end = time_decoder(decoded, batch_decoded)

    length_of_media = librosa.get_duration(y=audio_numpy, sr=16000)
    wcps = len(batch_decoded) / length_of_media

    # Make word list
    word_list, word_start, word_end = split_word_list(decoded, word_start, word_end)

    # Make fcpxml
    partial_fcpxml = partial(
        make_xml_from_words,
        word_start=word_start,
        word_end=word_end,
        wcps=wcps,
    )

    return partial_fcpxml, word_list


def add_captions(frame, word_generator):
    cv2.putText(
        frame,
        next(word_generator),
        (0, 200), # x, y
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255), # RGB
        3,
        cv2.LINE_AA,
    )
    return frame


def video_upload(project_name, uploaded_file=None, demo=False):

    # Convert the file to numpy.
    if uploaded_file != None:
        video_bytes = io.BytesIO(uploaded_file.read())
    else:
        video_bytes = None

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        if demo == True:
            cwd = Path(".")
            st.title("Before")

            file_finder = cwd.glob("*.mov")
            list_of_vids = [str(i) for i in file_finder]

            if "Success4.mov" not in list_of_vids:
                url = "https://drive.google.com/uc?id=1kUO0dKTsq4E2rFH1_JehUZC23giwVtY3"
                output = "Success4.mov"
                gdown.download(url, output, quiet=False)

            video_path = "Success4.mov"
            st.video(video_path)

        else:
            video_path = tmp_file.name
            fp = Path(video_path)
            fp.write_bytes(uploaded_file.getvalue())
            st.video(video_bytes)

        video = VideoFileClip(video_path)

        audio = video.audio
        duration = video.duration  # presented as seconds, float
        # note: video.fps != audio.fps

        new_audio = resample_numpy(audio.to_soundarray(), audio.fps)

        decoded, batch_decoded = transcribe_audio(new_audio)
        word_start, word_end = time_decoder(decoded, batch_decoded)

        length_of_media = video.duration
        wcps = len(batch_decoded) / length_of_media

        # Make word list
        word_list, word_start, word_end = split_word_list(decoded, word_start, word_end)
        partial_fcpxml = partial(
            make_xml_from_words,
            word_start=word_start,
            word_end=word_end,
            wcps=wcps,
        )

        new_text = st.text_area("Text (Edit to Change Video)", value="\n".join(word_list))
        new_word_list = new_text.splitlines()

        # convert word chunks into frames
        word_start_frames = [int(i / wcps * video.fps) for i in word_start]
        word_end_frames = [int(i / wcps * video.fps) for i in word_end]

        temp_list = []
        index = 0
        for text, start, end in zip(new_word_list, word_start_frames, word_end_frames):
            # check if space
            if start - index > 1:
                space_duration = start - index
                space_duration = [""] * space_duration
            else:
                space_duration = []

            duration = end - start + 1

            text_duration = [text] * duration

            temp_list.extend(space_duration + text_duration)
            index = end

        word_gen = word_generator(temp_list)
        partial_captions = partial(add_captions, word_generator=word_gen)

        if len(new_word_list) == len(word_list):
            out_video = video.fl_image(partial_captions)
            video_file_name = "temp.mp4"
            out_video.write_videofile(video_file_name, codec="libx264", fps=video.fps)
            st.title("After")
            st.video(video_file_name)

            text = partial_fcpxml(word_list=new_word_list)

            with open(video_file_name, "rb") as file_handler:
                btn = st.download_button(
                    label="Download Video file",
                    data=file_handler,
                    file_name=f"{project_name}.mp4",
                )

            btn = st.download_button(
                label="Download FCPX project file",
                data=text,
                file_name=f"{project_name}.fcpxml",
            )


def audio_upload(project_name, uploaded_file):
    file_bytes = io.BytesIO(uploaded_file.read())

    try:
        # Convert the file to numpy.
        audio_numpy = convert_audio_file(file_bytes)
        partial_fcpxml, word_list = process_audio(audio_numpy)

        new_text = st.text_area("Edit transcription text", value="\n".join(word_list))
        new_word_list = new_text.splitlines()

        if len(new_word_list) == len(word_list):
            text = partial_fcpxml(word_list=new_word_list)

            btn = st.download_button(
                label="Download FCPX project file",
                data=text,
                file_name=f"{project_name}.fcpxml",
            )

    except Exception as e:
        st.write(e)
        download_data(file_bytes)


def main():
    project_name = st.text_input("Project Name:", value="Project Name")

    uploaded_file = st.file_uploader("Upload audio/video file")

    if uploaded_file is not None:
        video_suffix_list = [
            "webm",
            "mkv",
            "flv",
            "vob",
            "ogv",
            "ogg",
            "drc",
            "avi",
            "mov",
            "qt",
            "wmv",
            "amv",
            "m4v",
            "svi",
            "f4v",
        ]

        if uploaded_file.name.split(".")[-1].lower() in video_suffix_list:
            video_upload(project_name, uploaded_file)
        else:
            try:
                audio_upload(project_name, uploaded_file)
            except Exception as e:
                st.write(e)

    else:
        video_upload(project_name, demo=True)


if __name__ == "__main__":
    main()
