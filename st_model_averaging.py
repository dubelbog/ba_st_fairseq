import os
import soundfile as sf
import torch
import torch.utils.data as data_utils

os.system("CHECKPOINT_FILENAME=avg_checkpoint_all.pt "
          "python /Users/bdubel/Documents/ZHAW/BA/fairseq/scripts/average_checkpoints.py "
          "--inputs /Users/bdubel/Documents/ZHAW/BA/data/swiss_all/checkpoints --num-epoch-checkpoints 14 "
          "--output /Users/bdubel/Documents/ZHAW/BA/data/swiss_all/checkpoints/avg_checkpoint_all.pt ")


# os.system("fairseq-generate /Users/bdubel/Documents/ZHAW/BA/data/covost/sv-SE "
#           "--config-yaml config_st_sv-SE_en.yaml --gen-subset test_st_sv-SE_en --task speech_to_text "
#           "--path /Users/bdubel/Documents/ZHAW/BA/data/covost/st_sv/checkpoint_best.pt "
#           "--max-tokens 50000 --beam 5 --scoring sacrebleu")
#
# wav_input_16khz = torch.randn(1, 10000)
# print(wav_input_16khz.data.type())
# audio_input, _ = sf.read("data/sound/en/m0003_us_m0003_00348.wav")
# print(audio_input.dtype)
# data_utils.TensorDataset(torch.from_numpy(audio_input).float(), torch.from_numpy(audio_input).double())
# print(audio_input.dtype)
# print("x")
# print(audio_input)
# x = torch.tensor([audio_input])
# print(x.data.type())
# x = x.type(torch.FloatTensor)
# print(x.data.type())


