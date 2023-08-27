cd /path/to/TVTS/v2

sudo chmod -R 777 .

python -m torch.distributed.launch $@ \
  --nnodes 1 \
  --nproc_per_node 8 \
  --node_rank 0 \
  --master_port 12320 \
  --use_env \
  train_dist_TVTSv2_ViT_B_16.py \
  --config configs/dist-yt-web-pt-vit-b-16.json \
  --schedule 6 8

if [ $? != 0 ]; then
  echo "Fail! Exit with 1"
  exit 1
else
  echo "Success! Exit with 0"
  exit 0
fi
