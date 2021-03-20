import soundfile as sf
import torch
import torch.utils.data as data_utils

# sound = AudioSegment.from_mp3('data/sound/de/common_voice_de_17299209.mp3')
# sound.export('data/sound/de/converted_to_wav_file.wav', format="wav")
# audio_input_de, _ = sf.read("data/sound/en/f0001_us_f0001_00001.wav")
# os.system("afplay data/sound/de/converted_to_wav_file.wav")

# audio_input, _ = sf.read("data/sound/en/f0001_us_f0001_00143.wav")
# print(len(audio_input))
# print(type(audio_input))
# audio_torch = torch.from_numpy(audio_input)
# print(type(audio_torch))
#
# print(audio_torch.size())
#
# wav_input_16khz = torch.randn(1, 10000)
# print(type(wav_input_16khz))
# print(wav_input_16khz.size())


wav_input_16khz = torch.randn(1, 10000)
print(wav_input_16khz.data.type())
audio_input, _ = sf.read("data/sound/en/m0003_us_m0003_00348.wav")
print(audio_input.dtype)
data_utils.TensorDataset(torch.from_numpy(audio_input).float(), torch.from_numpy(audio_input).double())
print(audio_input.dtype)
print("x")
print(audio_input)
x = torch.tensor([audio_input])
print(x.data.type())
x = x.type(torch.FloatTensor)
print(x.data.type())












