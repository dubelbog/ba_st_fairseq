import soundfile as sf
import torch
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer

# load pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")

# load audio
audio_input, _ = sf.read("data/sound/es/clf_00610_00129984119.wav")

# transcribe
input_values = tokenizer(audio_input, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]
print(transcription.lower())
