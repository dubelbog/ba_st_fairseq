import os

# os.system("CHECKPOINT_FILENAME=avg_last_4_checkpoint.pt "
#           "python ../scripts/average_checkpoints.py "
#           "--inputs /Users/bdubel/Documents/ZHAW/BA/data/covost/st_sv --num-epoch-checkpoints 4 "
#           "--output /Users/bdubel/Documents/ZHAW/BA/data/covost/st_sv/checkpoint_best.pt ")


os.system("fairseq-generate /Users/bdubel/Documents/ZHAW/BA/data/covost/sv-SE "
          "--config-yaml config_st_sv-SE_en.yaml --gen-subset test_st_sv-SE_en --task speech_to_text "
          "--path /Users/bdubel/Documents/ZHAW/BA/data/covost/st_sv/checkpoint_best.pt "
          "--max-tokens 50000 --beam 5 --scoring sacrebleu")

