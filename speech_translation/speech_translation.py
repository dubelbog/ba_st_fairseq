import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os

#
# List available models
torch.hub.list('pytorch/fairseq', force_reload=True)  # [..., 'transformer.wmt16.en-de', ... ]

# Load a transformer trained on WMT'16 En-De
# Note: WMT'19 models use fastBPE instead of subword_nmt, see instructions below
# en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de',
#                        tokenizer='moses', bpe='subword_nmt')

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
en2de.eval()  # disable dropout


# load pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# load audio
# example of translation
# audio_input, _ = sf.read("data/sound/en/m0003_us_m0003_00348.wav")
# os.system("afplay data/sound/en/m0003_us_m0003_00348.wav")

# example of translation
# audio_input, _ = sf.read("data/sound/en/m0004_us_m0004_00148.wav")
# os.system("afplay data/sound/en/m0004_us_m0004_00148.wav")

# simple example
# f0001_us_f0001_00001.wav

audio_input, _ = sf.read("data/sound/en/f0001_us_f0001_00001.wav")
os.system("afplay data/sound/en/f0001_us_f0001_00001.wav")

# transcribe
input_values = tokenizer(audio_input, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]
print("Transcription")
print(transcription.lower())
print("Translation")
print(en2de.translate(transcription.lower()))

