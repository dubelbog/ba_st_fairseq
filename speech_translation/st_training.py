import os

os.system("../fairseq_cli/train.py /Users/bdubel/Documents/ZHAW/BA/data/covost/sv-SE "
          "--config-yaml config_st_sv-SE_en.yaml --train-subset train_st_sv-SE_en --valid-subset dev_st_sv-SE_en "
          "--save-dir /Users/bdubel/Documents/ZHAW/BA/data/covost/st_sv "
          "--num-workers 4 --max-tokens 40000 --max-update 60000 --task speech_to_text "
          "--criterion label_smoothed_cross_entropy --report-accuracy --arch s2t_transformer_s --optimizer adam "
          "--lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8")
# "--load-pretrained-encoder-from /Users/bdubel/Documents/ZHAW/BA/data/covost/covost2_en_asr_transformer_s.pt")
