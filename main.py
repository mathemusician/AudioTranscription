import io
import torch
import gdown
import base64
import librosa
import difflib
import soundfile
import numpy as np
import pickle as pl
import streamlit as st
from pathlib import Path
from copy import deepcopy
import scipy.signal as sps
from functools import partial
from pydub import AudioSegment
from multipage import MultiPage
from consolidate import time_decoder
from split_texts import split_by_words
from make_xml import make_xml_from_words
from transformers import Wav2Vec2Processor
from moviepy.video.tools.subtitles import SubtitlesClip
from flash.core.data.data_source import DefaultDataKeys
from flash.audio import SpeechRecognition, SpeechRecognitionData
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


# stop multiprocessing
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.experimental_singleton
def model_and_processor():
    # load model and processor once
    backbone = "facebook/wav2vec2-base-960h" # "facebook/wav2vec2-base-960h" #patrickvonplaten/wav2vec2_tiny_random_robust
    model = SpeechRecognition(backbone)  # Wav2Vec2ForCTC.from_pretrained(backbone)
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
    new_rate = 16000
    number_of_samples = round(len(audio_numpy) * float(new_rate) / sample_rate)
    resampled_audio = sps.resample(audio_numpy, number_of_samples)
    return resampled_audio


def convert_audio_file(audio_bytes) -> None:
    try:
        clip, sample_rate = soundfile.read(audio_bytes)
    except RuntimeError:
        audio_bytes.seek(0)
        audio_object = AudioSegment.from_file_using_temporary_files(audio_bytes)
        clip = np.array(audio_object.get_array_of_samples())
        sample_rate = audio_object.frame_rate
    resampled_audio = resample_numpy(clip, sample_rate)
    return resampled_audio


def transcribe_audio(audio_numpy):
    """custom_datamodule = SpeechRecognitionData.from_json(
        input_fields="file", target_fields="text", test_file="text.json"
    )"""
    # custom_datamodule = SpeechRecognitionData.from_numpy(test_data=audio_numpy)
    try:
        if audio_numpy.shape[0] > audio_numpy.shape[1]:
            audio_numpy = audio_numpy.transpose()
    except IndexError:
        pass

    input_ = processor(librosa.to_mono(audio_numpy))
    audio_dict = {
        DefaultDataKeys.INPUT: input_["input_values"][0],
        DefaultDataKeys.TARGET: "dummy target",
        DefaultDataKeys.METADATA: {"sampling_rate": 16000},
    }

    # predictions = model.predict([custom_datamodule._test_ds[0]])

    predictions = model.predict([audio_dict])

    pred_ids = torch.argmax(predictions[0].logits, dim=-1)

    decoded = processor.decode(pred_ids)
    batch_decoded = processor.batch_decode(pred_ids)

    return decoded, batch_decoded


def split_word_list(decoded, word_start, word_end):
    word_list = decoded.split()
    old_word_list = deepcopy(word_list)

    # Make sure length of words are the same
    assert len(old_word_list) == len(word_list)
    word_list, word_start, word_end = split_by_words(
        word_list=word_list, word_start=word_start, word_end=word_end
    )

    return word_list, word_start, word_end


def parse_text_box():
    # use fuzzy matching to match rows?
    # use Git to compare lines?
    pass


def text_clip(text: str):
        """
        Return a description string on the bottom-left of the video

        Args:
                    text (str): Text to show

        Returns:
                    moviepy.editor.TextClip: A instance of a TextClip
        """
        my_text = (
            TextClip(text, font="Helvetica", fontsize=50, color="white")
            .set_position(("center", "center"))
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
    fcpxml_func = partial(make_xml_from_words,
        word_start=word_start,
        word_end=word_end,
        wcps=wcps,
    )

    return fcpxml_func, word_list


def demo_video():
    project_name = st.text_input("Project Name:", value="Project Name")

    cwd = Path(".")
    st.title("Audio Transcription")

    file_finder = cwd.glob("*.mov")
    list_of_vids = [str(i) for i in file_finder]

    if "Success4.mov" not in list_of_vids:
        url = "https://drive.google.com/uc?id=1kUO0dKTsq4E2rFH1_JehUZC23giwVtY3"
        output = "Success4.mov"
        gdown.download(url, output, quiet=False)
    
    video_path = "Success4.mov"
    video_file = open(video_path, "rb")
    video_bytes = video_file.read()

    st.video(video_bytes)

    video = VideoFileClip(video_path)

    audio = video.audio
    duration = video.duration  # presented as seconds, float
    # note video.fps != audio.fps

    new_audio = resample_numpy(audio.to_soundarray(), audio.fps)


    decoded, batch_decoded = transcribe_audio(new_audio)
    word_start, word_end = time_decoder(decoded, batch_decoded)

    length_of_media = video.duration
    wcps = len(batch_decoded) / length_of_media

    # Make word list
    word_list, word_start, word_end = split_word_list(decoded, word_start, word_end)

    fcpxml_func = partial(make_xml_from_words,
        word_start=word_start,
        word_end=word_end,
        wcps=wcps,
    )

    new_text = st.text_area('Text', value='\n'.join(word_list))
    new_word_list = new_text.splitlines()

    if len(new_word_list) == len(word_list):
        text = fcpxml_func(word_list=new_word_list)
        
        btn = st.download_button(
            label="Download FCPX project file",
            data=text,
            file_name=f"{project_name}.fcpxml",
        )



def demo_audio():
    pass

def video_upload():
    project_name = st.text_input("Project Name:", value="Project Name")

    uploaded_file = st.file_uploader("Choose an audio file")


    if uploaded_file is not None:
        # Convert the file to numpy.
        video_bytes = io.BytesIO(uploaded_file.read())
        # video_bytes = video_file.read()

        st.video(video_bytes)

        video = VideoFileClip(video_path)

        audio = video.audio
        duration = video.duration  # presented as seconds, float
        # note video.fps != audio.fps

        new_audio = resample_numpy(audio.to_soundarray(), audio.fps)


        decoded, batch_decoded = transcribe_audio(new_audio)
        word_start, word_end = time_decoder(decoded, batch_decoded)

        length_of_media = video.duration
        wcps = len(batch_decoded) / length_of_media

        # Make word list
        word_list, word_start, word_end = split_word_list(decoded, word_start, word_end)

        fcpxml_func = partial(make_xml_from_words,
            word_start=word_start,
            word_end=word_end,
            wcps=wcps,
        )

        new_text = st.text_area('Text', value='\n'.join(word_list))
        new_word_list = new_text.splitlines()

        if len(new_word_list) == len(word_list):
            text = fcpxml_func(word_list=new_word_list)
            
            btn = st.download_button(
                label="Download FCPX project file",
                data=text,
                file_name=f"{project_name}.fcpxml",
            )


def audio_upload():
    project_name = st.text_input("Project Name:", value="Project Name")

    uploaded_file = st.file_uploader("Choose an audio file")


    if uploaded_file is not None:
        # Convert the file to numpy.
        file_bytes = io.BytesIO(uploaded_file.read())
        
        try:
            audio_numpy = convert_audio_file(file_bytes)
            fcpxml_func, word_list = process_audio(audio_numpy)

            new_text = st.text_area('Text', value='\n'.join(word_list))
            new_word_list = new_text.splitlines()

            if len(new_word_list) == len(word_list):
                text = fcpxml_func(word_list=new_word_list)
                
                btn = st.download_button(
                    label="Download FCPX project file",
                    data=text,
                    file_name=f"{project_name}.fcpxml",
                )

        except Exception as e:
            st.write(e)
            download_data(file_bytes)



def main():
    app = MultiPage()

    st.title("Audio Transcription")

    app.add_page("Demo Video", demo_video)
    app.add_page("Video Upload", video_upload)
    app.add_page("Upload Audio", audio_upload)
    
    app.run()
    
    
    

if __name__ == "__main__":
    main()
