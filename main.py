import torch
import librosa
import difflib
import soundfile
import numpy as np
import pickle as pl
import streamlit as st
from transformers import Wav2Vec2Processor
from pathlib import Path
from copy import deepcopy
from consolidate import time_decoder
from split_texts import split_by_words
from make_xml import make_xml_from_words
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from flash.audio import SpeechRecognition, SpeechRecognitionData
import io
import scipy.signal as sps
from flash.core.data.data_source import DefaultDataKeys
#from data import SpeechRecognitionData
# from transformers import Wav2Vec2


@st.cache(persist=True)
def model_and_processor():
    # load model and processor once
    backbone = "facebook/wav2vec2-base-960h"
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


def convert_audio_file(audio_bytes) -> None:
    new_rate = 16000
    clip, sample_rate = soundfile.read(audio_bytes)
    number_of_samples = round(len(clip) * float(new_rate) / sample_rate)
    resampled_audio = sps.resample(clip, number_of_samples)
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


def make_fcpxml(word_start, word_end, word_list, wcps):
    """
    with open("test.fcpxml", "w") as file_handler:
        file_handler.write(
            make_xml_from_words(
                word_start=word_start,
                word_end=word_end,
                word_list=word_list,
                wcps=wcps,
            )
        )
    """


def process_audio(audio_numpy):
    decoded, batch_decoded = transcribe_audio(audio_numpy)
    word_start, word_end = time_decoder(decoded, batch_decoded)

    length_of_media = librosa.get_duration(*librosa.load("Test3.wav"))  # 12.11 seconds
    wcps = len(batch_decoded) / length_of_media

    # Make word list
    word_list, word_start, word_end = split_word_list(decoded, word_start, word_end)

    # Make fcpxml
    # return make_fcpxml(word_start, word_end, word_list, wcps)
    return make_xml_from_words(
        word_start=word_start,
        word_end=word_end,
        word_list=word_list,
        wcps=wcps,
    )


def main():
    st.title("Audio Transcription")
    
    project_name = st.text_input("Project Name:", value="Project Name")

    uploaded_file = st.file_uploader("Choose an audio file")

    if uploaded_file is not None:
        # Convert the file to numpy.
        file_bytes = io.BytesIO(uploaded_file.read())
        audio_numpy = convert_audio_file(file_bytes)
        text = process_audio(audio_numpy)

        btn = st.download_button(
            label="Download FCPX project file",
            data=file,
            file_name=f"{project_name}.fcpxml",
        )
    

if __name__ == "__main__":
    # parse_text_box()
    main()
