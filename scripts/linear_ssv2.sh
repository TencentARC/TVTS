cd /path/to/TVTS/downstream

PATH=$(readlink -f ffmpeg-release):$PATH

export NCCL_DEBUG=INFO

NOW="$(date +%Y%m%d%H%M%S)"
JOB_NAME=TVTS_linear_ssv2

sudo chmod -R 777 ..

# Set the path to save checkpoints
OUTPUT_DIR='../results/TVTS_downstream/linear_ssv2'
# path to SSV2 set (train.csv/val.csv/test.csv)
DATA_PATH='data/SSV2'
# path to pretrain model
MODEL_PATH='../TVTS_yt_pt.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
python -m torch.distributed.launch $@ run_class_linear.py \
  --model vit_base_patch16_224 \
  --data_set SSV2 \
  --nb_classes 174 \
  --data_path ${DATA_PATH} \
  --finetune ${MODEL_PATH} \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 12 \
  --num_sample 1 \
  --input_size 224 \
  --short_side_size 224 \
  --save_ckpt_freq 10 \
  --num_frames 16 \
  --opt sgd \
  --lr 0.1 \
  --weight_decay 1e-9 \
  --momentum 0.9 \
  --warmup_epochs 10 \
  --epochs 100 \
  --dist_eval \
  --test_num_segment 2 \
  --test_num_crop 3 \
  --enable_deepspeed

if [ $? != 0 ]; then
  echo "Fail! Exit with 1"
  exit 1
else
  echo "Success! Exit with 0"
  exit 0
fi
