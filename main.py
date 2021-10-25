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
#from data import SpeechRecognitionData
# from transformers import Wav2Vec2


@st.cache(persist=True)
def model_and_processor():
    # load model and processor once
    backbone = "facebook/wav2vec2-base-960h"
    model = Wav2Vec2ForCTC.from_pretrained(backbone)
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
    audio_file = Path("/Users/mosaicchurchhtx/Desktop/Untitled.m4a")
    # x, sr = librosa.load(audio_file, sr=16000)
    x, sr = soundfile.read(audio_bytes)
    resampled_audio = librosa.resample(x, sr, 16000)
    # soundfile.write("Test3.wav", x, samplerate=16000)
    return resampled_audio


def transcribe_audio(audio_numpy):
    """custom_datamodule = SpeechRecognitionData.from_json(
        input_fields="file", target_fields="text", test_file="text.json"
    )"""
    custom_datamodule = SpeechRecognitionData.from_numpy(test_data=audio_numpy)

    predictions = model.predict([custom_datamodule._test_ds[0]])

    pred_ids = torch.argmax(torch.stack(predictions[0].logits), dim=-1)

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
        file_bytes = np.asarray(bytearray(uploaded_file.read()))
        print(uploaded_file.read())
        print(bytearray(uploaded_file.read()))
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
