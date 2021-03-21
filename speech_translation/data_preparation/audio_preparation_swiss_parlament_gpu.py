from pydub import AudioSegment
import torchaudio
from examples.speech_to_text.data_utils import extract_fbank_features, save_df_to_tsv, gen_config_yaml, gen_vocab
from pathlib import Path
import pandas as pd
import sys
sys.path.append('/cluster/home/dubelbog/tools/python3_env/lib/python3.8/site-packages/ffmpeg')

root_path = "../data/swiss_corpus/"
path_manifest_swiss = root_path + "test_sample.tsv"
clip_path = root_path + "clips_sample/"
mp3_path = root_path + "mp3/"
feature_root = Path("../data/sound").absolute() / "fbank"
suffix_flac = ".flac"
suffix_mp3 = ".mp3"
ms = 1000

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def print_audio_infos(track):
    print("frame_rate", track.frame_rate)
    print("duration_seconds", track.duration_seconds)
    print("sample_width", track.sample_width)
    print("frame_width", track.frame_width)
    print("frame_count", track.frame_count())


def text_processing(data):
    tgt_text_arr = data[2:len(data)-1]
    tgt_text = " ".join(tgt_text_arr)
    tgt_text = tgt_text.replace('ß', 'ss')
    tgt_text = tgt_text[0].lower() + tgt_text[1: len(tgt_text)]
    replace_signs = ['-', '–', "»", "«", ".", ","]
    for char in replace_signs:
        tgt_text = tgt_text.replace(char, "")
    return tgt_text


def manifest_preparation(manifest, track, data, tgt_text, track_path):
    waveform, sample_rate = torchaudio.load(track_path)
    utt_id = data[1].removesuffix(".flac")
    extract_fbank_features(waveform, sample_rate, feature_root / f"{utt_id}.npy")
    manifest["id"].append(utt_id)
    manifest["audio"].append(feature_root / f"{utt_id}.npy")
    duration_ms = track.duration_seconds * ms
    manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
    manifest["tgt_text"].append(tgt_text)
    manifest["speaker"].append(data[0])


def audio_processing(data, manifest, tgt_text):
    file = data[1]
    track_path = mp3_path + file + suffix_mp3
    audio_file = clip_path + file
    audio_file = AudioSegment.from_file(audio_file)
    audio_file.export(track_path)
    track = AudioSegment.from_file(track_path)
    track.export(track_path)
    manifest_preparation(manifest, track, data, tgt_text, track_path)


def gen_voc(train_text, spm_filename_prefix):
    f = open(Path("../data/sound").absolute() / "test.txt", "a")
    for t in train_text:
        f.write(" ".join(t.split()[0:4]) + "\n")
    print(f.name)
    gen_vocab(
        Path(f.name),
        Path("../data/sound") / spm_filename_prefix
    )


def preparation():
    manifest_swiss = open(path_manifest_swiss, "r")
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    train_text = []
    counter = 0
    for line in manifest_swiss:
        if counter != 0:
            data = line.split()
            tgt_text = text_processing(data)
            print(tgt_text)
            train_text.append(tgt_text)
            audio_processing(data, manifest, tgt_text)
        counter = counter + 1
        # generate manifest
    df = pd.DataFrame.from_dict(manifest)
    split = "dev"
    task = "st_ch_de"
    save_df_to_tsv(df, Path("../data/sound") / f"{split}_{task}.tsv")
    spm_filename_prefix = f"spm_char_{task}"
    # Generate config YAML
    gen_config_yaml(
        Path("../data/sound"),
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{task}.yaml",
        specaugment_policy="lb",
        )
    # generating vocabulary
    if len(train_text) > 0:
        gen_voc(train_text, spm_filename_prefix)


# def prepare_audio_files():
#     stop = 1
#     counter = 1
#     manifest = {c: [] for c in MANIFEST_COLUMNS}
#     translation = open(path_manifest_swiss, "r")
#     train_text = []
#     for line, tr in translation:
#         data = line.split()
#         track_path = root_path + "segments/" + data[0] + suffix_mp3
#         if os.path.isfile(track_path):
#             counter = counter + 1
#         else:
#             counter = 1
#             audio_file = prefix + data[0] + suffix_flac
#             audio_file = AudioSegment.from_file(audio_file)
#             audio_file.export(track_path)
#
#         track = AudioSegment.from_file(track_path)
#         process_data(data, counter, track, manifest, tr.rstrip())
#         train_text.extend(tr.rstrip())
#         stop = stop + 1
#         if stop > 10:
#             df = pd.DataFrame.from_dict(manifest)
#             split = "dev"
#             task = "st"
#             save_df_to_tsv(df, Path("../data/sound") / f"{split}_{task}.tsv")
#
#             spm_filename_prefix = f"spm_char_{task}"
#             # with NamedTemporaryFile(mode="w") as f:
#             #     for t in train_text:
#             #         f.write(t + "\n")
#             #     gen_vocab(
#             #         Path(f.name),
#             #         Path("../speech_translation/data/sound") / spm_filename_prefix
#             #     )
#             # Generate config YAML
#             gen_config_yaml(
#                 Path("../data/sound"),
#                 spm_filename_prefix + ".model",
#                 yaml_filename=f"config_{task}.yaml",
#                 specaugment_policy="lb",
#                 )
#             return


preparation()

# m4a_audio = \
#     AudioSegment.from_file("/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/clips/en.20080924.31.3-260.m4a")
# m4a_audio.export("/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/segments/test.mp3",
#                  format="mp3")
#
# song = AudioSegment.from_file("/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/segments/test.mp3")
#
# interval_1 = 8.95 * 1000
#
# path = "/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/segments/first_track.mp3"
# first_track = song[:interval_1]
# first_track.export(path, format="mp3")
#
# print("frame_rate", first_track.frame_rate)
# print("duration_seconds", first_track.duration_seconds)
# print("sample_width", first_track.sample_width)
# print("frame_width", first_track.frame_width)
# print("frame_count", first_track.frame_count())
# waveform_1, sample_rate_1 = torchaudio.load(path)
# print(waveform_1)
# print(sample_rate_1)
# print(int(1 + ((first_track.duration_seconds*1000) - 25) / 10))
#
# first_track = first_track.set_frame_rate(frame_rate=48000)
# first_track.export("/Users/bdubel/Documents/ZHAW/BA/data/europarl_st/pt_es/segments/first_track_1.mp3",
#                    format="mp3")
# print("AFTER")
# print("frame_rate", first_track.frame_rate)
# print("duration_seconds", first_track.duration_seconds)

