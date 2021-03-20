from pydub import AudioSegment
import torchaudio
import os.path
from examples.speech_to_text.data_utils import extract_fbank_features, save_df_to_tsv, gen_config_yaml, gen_vocab
from pathlib import Path
import pandas as pd

root_path = "/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/"
path_segments = root_path + "es/dev/segments.lst"
path_translations = root_path + "es/dev/segments.es"
prefix = root_path + "clips/"
suffix_m4a = ".m4a"
suffix_mp3 = ".mp3"
ms = 1000

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def add_data_to_manifest(manifest, track_path_segment, data, counter, track_segment, tr):
    waveform, sample_rate = torchaudio.load(track_path_segment)
    feature_root = Path("../speech_translation/data/sound").absolute() / "fbank"
    utt_id = data[0] + "_" + str(counter)
    extract_fbank_features(waveform, sample_rate, feature_root / f"{utt_id}.npy")
    manifest["id"].append(utt_id)
    manifest["audio"].append(feature_root / f"{utt_id}.npy")
    duration_ms = track_segment.duration_seconds * ms
    manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
    manifest["tgt_text"].append(tr)
    manifest["speaker"].append(21918)


def process_data(data, counter, track, manifest, tr):
    start = float(data[1]) * ms
    end = float(data[2]) * ms
    track_segment = track[start:end]
    track_segment = track_segment.set_frame_rate(frame_rate=48000)
    track_path_segment = root_path + "segments/" + data[0] + "_" + str(counter) + suffix_mp3
    track_segment.export(track_path_segment)
    add_data_to_manifest(manifest, track_path_segment, data, counter, track_segment, tr)


def prepare_audio_files():
    stop = 1
    counter = 1
    f = open(path_segments, "r")
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    translation = open(path_translations, "r")
    train_text = []
    for line, tr in zip(f, translation):
        data = line.split()
        track_path = root_path + "segments/" + data[0] + suffix_mp3
        if os.path.isfile(track_path):
            counter = counter + 1
        else:
            counter = 1
            audio_file = prefix + data[0] + suffix_m4a
            audio_file = AudioSegment.from_file(audio_file)
            audio_file.export(track_path)

        track = AudioSegment.from_file(track_path)
        process_data(data, counter, track, manifest, tr.rstrip())
        train_text.extend(tr.rstrip())
        stop = stop + 1
        if stop > 10:
            df = pd.DataFrame.from_dict(manifest)
            split = "dev"
            task = "st"
            save_df_to_tsv(df, Path("../speech_translation/data/sound") / f"{split}_{task}.tsv")

            spm_filename_prefix = f"spm_char_{task}"
            # with NamedTemporaryFile(mode="w") as f:
            #     for t in train_text:
            #         f.write(t + "\n")
            #     gen_vocab(
            #         Path(f.name),
            #         Path("../speech_translation/data/sound") / spm_filename_prefix
            #     )
            # Generate config YAML
            gen_config_yaml(
                Path("../speech_translation/data/sound"),
                spm_filename_prefix + ".model",
                yaml_filename=f"config_{task}.yaml",
                specaugment_policy="lb",
                )
            return


prepare_audio_files()

m4a_audio = \
    AudioSegment.from_file("/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/clips/en.20080924.31.3-260.m4a")
m4a_audio.export("/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/segments/test.mp3",
                 format="mp3")

song = AudioSegment.from_file("/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/segments/test.mp3")

interval_1 = 8.95 * 1000

path = "/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/segments/first_track.mp3"
first_track = song[:interval_1]
first_track.export(path, format="mp3")

print("frame_rate", first_track.frame_rate)
print("duration_seconds", first_track.duration_seconds)
print("sample_width", first_track.sample_width)
print("frame_width", first_track.frame_width)
print("frame_count", first_track.frame_count())
waveform_1, sample_rate_1 = torchaudio.load(path)
print(waveform_1)
print(sample_rate_1)
print(int(1 + ((first_track.duration_seconds*1000) - 25) / 10))

first_track = first_track.set_frame_rate(frame_rate=48000)
first_track.export("/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/segments/first_track_1.mp3",
                   format="mp3")
print("AFTER")
print("frame_rate", first_track.frame_rate)
print("duration_seconds", first_track.duration_seconds)

