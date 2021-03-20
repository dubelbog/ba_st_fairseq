import os

os.system("../fairseq_cli/train.py /Users/bdubel/Documents/ZHAW/BA/data/covost/es "
          "--config-yaml config_asr_es.yaml --train-subset train_asr_es --valid-subset dev_asr_es "
          "--save-dir /Users/bdubel/Documents/ZHAW/BA/data/covost/asr_es "
          "--num-workers 4 --max-tokens 40000 --max-update 60000 --task speech_to_text "
          "--criterion label_smoothed_cross_entropy --report-accuracy --arch s2t_transformer_s --optimizer adam "
          "--lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8")

