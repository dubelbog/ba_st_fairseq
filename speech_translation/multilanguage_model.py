import torch
import soundfile as sf
import fairseq
import numpy as np
from itertools import groupby


cp_path = 'data/model/xlsr_53_56k.pt'
# cp_path = 'data/model/wav2vec_small.pt' model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
# cp_path]) model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path], arg_overrides={
# "data": "data/model/dictionary.rtf" })
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()


# wav_input_16khz = torch.randn(1, 10000)
# audio_input, _ = sf.read("data/sound/en/m0003_us_m0003_00348.wav")
audio_input, _ = sf.read("data/sound/en/f0001_us_f0001_00001.wav")
audio_input = torch.tensor([audio_input])
audio_input = audio_input.type(torch.FloatTensor)
print(audio_input.size())


# todo input Parameter: torch.FloatTensor with size [1, n]
kl = model.forward(audio_input)

z = model.feature_extractor(audio_input)
# c = model.feature_aggregator(z)

# decode the prediction
# I think those "logits" you extracted in there aren't actually logits, but just output features
# todo: size [101, 1, n] => n depends on input
logits = model(source=audio_input, padding_mask=None)["x"]

print("Length: ", len(logits.detach().numpy()))
print("Size: ", logits.size())
torch.transpose(logits, 0, 2)[0].size()
# torch.transpose(input, dim0, dim1)
# for i in logits.detach().numpy():
#     print(i)
#     print("length 2: ", len(i))

predicted_ids = torch.argmax(logits[:, 0], axis=-1)

json_dict = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10,
             "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22,
             "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}

print(predicted_ids)
look_up = np.asarray(list(json_dict.keys()))
# print(look_up)
# converted_tokens = look_up[predicted_ids]
# fused_tokens = [tok[0] for tok in groupby(converted_tokens)]
# output = ' '.join(''.join(''.join(fused_tokens).split("<s>")).split("|"))
#
#
# print("Prediction: ", output)

# input_values = tokenizer(audio_input, return_tensors="pt").input_values
# logits = model(input_values).logits
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = tokenizer.batch_decode(predicted_ids)[0]
# print(c)

