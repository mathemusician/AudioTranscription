import librosa
import soundfile
from pathed import Path

audio_file = Path("/Users/mosaicchurchhtx/Desktop/Ableton.wav") #48KHz

x, sr = librosa.load(audio_file, sr=44100)
y = librosa.resample(x, 44100, 16000)
soundfile.write("Test3.wav", y, samplerate=16000)
