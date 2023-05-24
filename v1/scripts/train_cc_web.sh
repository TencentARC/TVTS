cd /path/to/TVTS

export NCCL_DEBUG=INFO

NOW="$(date +%Y%m%d%H%M%S)"
JOB_NAME=TVTS-dist-cc-web-pt

sudo chmod -R 777 .

python -m torch.distributed.launch $@ train_dist_TVTS.py \
  --config configs/dist-cc-web-pt.json

if [ $? != 0 ]; then
  echo "Fail! Exit with 1"
  exit 1
else
  echo "Success! Exit with 0"
  exit 0
fi
