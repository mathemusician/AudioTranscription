import flash
import torch
import librosa
import soundfile
from transformers import Wav2Vec2Processor
from flash.audio import SpeechRecognition, SpeechRecognitionData
from fastpunct import FastPunct
from copy import deepcopy
from pathed import Path
from split_texts import split_by_words
from consolidate import time_decoder
from make_xml import make_xml_from_words
import pickle as pl

def pickle(data, filename="data.pickle"):
    with open(filename, "wb") as file_handler:
        pl.dump(data, file_handler)

def unpickle(filename="data.pickle"):
    with open(filename, "rb") as file_handler:
        return pl.load(file_handler)


if __name__ == "__main__":

    # convert audio file
    audio_file = Path("/Users/mosaicchurchhtx/Desktop/Untitled.m4a")
    x, sr = librosa.load(audio_file, sr=16000)
    soundfile.write("Test3.wav", x, samplerate=16000)

    custom_datamodule = SpeechRecognitionData.from_json(
        input_fields="file", target_fields="text", test_file="text.json"
    )

    model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    predictions = model.predict([custom_datamodule._test_ds[0]])

    pred_ids = torch.argmax(torch.stack(predictions[0].logits), dim=-1)
    # use in tandem with batch_decode to find time!!!

    decoded = processor.decode(pred_ids)
    batch_decoded = processor.batch_decode(pred_ids)

    word_start, word_end = time_decoder(decoded, batch_decoded)

    length_of_media = librosa.get_duration(*librosa.load("Test3.wav"))  # 12.11 seconds
    wcps = len(batch_decoded) / length_of_media


    # Add punctutation, fix spelling errors
    pickle(decoded)
    """
    fastpunct = FastPunct()
    decoded = decoded.lower()
    decoded = decoded.split()
    n = 500  # split into word groups of 500
    decoded = [" ".join(decoded[i : i + n]) for i in range(0, len(decoded), n)]
    decoded = fastpunct.punct(decoded, correct=True)
    decoded = " ".join(decoded)
    """

    # Make word list
    word_list = decoded.split()
    old_word_list = deepcopy(word_list)

    # Make sure length of words are the same
    assert len(old_word_list) == len(word_list)
    # pickle([word_list, word_start, word_end])
    word_list, word_start, word_end = split_by_words(
        word_list=word_list, word_start=word_start, word_end=word_end
    )

    with open("test.fcpxml", "w") as file_handler:
        file_handler.write(
            make_xml_from_words(
                word_start=word_start,
                word_end=word_end,
                word_list=word_list,
                wcps=wcps,
            )
        )
