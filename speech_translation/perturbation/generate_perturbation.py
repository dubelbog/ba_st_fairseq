import torch
from torch_audiomentations import Compose, Gain, PolarityInversion
import torchaudio

# Initialize augmentation callable
apply_augmentation = Compose(
    transforms=[
        Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=40.0,
            p=1.0,
        ),
        PolarityInversion(p=0.0)
    ]
)

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make an example tensor with white noise.
# This tensor represents 8 audio snippets with 2 channels (stereo) and 2 s of 16 kHz audio.
# audio_samples = torch.rand(size=(8, 2, 32000), dtype=torch.float32, device=torch_device) - 0.5
audio_samples = torchaudio.load("/Users/bdubel/Documents/ZHAW/BA/data/eth_ch_dialects/ag/ch_ag_0107.wav")

# Apply augmentation. This varies the gain and polarity of (some of)
# the audio snippets in the batch independently.
perturbed_audio_samples = apply_augmentation(audio_samples[0], sample_rate=16000)
torchaudio.save('/Users/bdubel/Documents/ZHAW/BA/data/swiss_all/perturbation/test1.flac',
                perturbed_audio_samples, sample_rate=16000)

