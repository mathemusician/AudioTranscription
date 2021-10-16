import flash
import torch
from transformers import Wav2Vec2Processor
from flash.audio import SpeechRecognition, SpeechRecognitionData


datamodule = SpeechRecognitionData.from_json(
    input_fields="file",
    target_fields="text",
    train_file="data/timit/train.json",
    test_file="data/timit/test.json",
)

custom_datamodule = SpeechRecognitionData.from_json(
    input_fields="file",
    target_fields="text",
    test_file="text.json"
)


model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


predictions = model.predict([custom_datamodule._test_ds[0]])


pred_ids = torch.argmax(torch.stack(predictions[0].logits), dim=-1)
# use in tandem with batch_decode to find time!!!
print(processor.decode(pred_ids))

