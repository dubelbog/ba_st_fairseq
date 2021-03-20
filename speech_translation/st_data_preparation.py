import os


def start_preparation_st(test_data_path, data_root, src_lang, tgt_lang):
    os.system(test_data_path +
              " --data-root " + data_root +
              " --vocab-type char"
              " --src-lang " + src_lang +
              " --tgt-lang " + tgt_lang)


def start_preparation_asr(test_data_path, data_root, src_lang):
    os.system(test_data_path +
              " --data-root " + data_root +
              " --vocab-type char"
              " --src-lang " + src_lang)


test_data_path_covost = "../examples/speech_to_text/prep_covost_data.py"
data_root_prefix = "/Users/bdubel/Documents/ZHAW/BA/data/"
data_root_covost = data_root_prefix + "covost"
src_lang_sv = "sv-SE"
tgt_lang_en = "en"


start_preparation_st(test_data_path_covost, data_root_covost, src_lang_sv, tgt_lang_en)

