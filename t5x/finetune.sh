MODEL_DIR="/tmp/finetune-model/"
python -m t5x.train \
  --gin_file=counsel_bot_finetune.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --alsologtostderr