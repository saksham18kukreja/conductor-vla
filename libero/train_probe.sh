#!/bin/sh
# Cost probe for fine-tuning SmolVLA on this laptop (RTX 4060, 8 GB; 15 GB RAM).
# Short lerobot-train runs from lerobot/smolvla_libero on lerobot/libero, its
# own recipe (camera rename, lr 1e-4, cosine), with GPU memory sampled every
# 2 s. Nothing from these runs is kept as a model.
#   usage: train_probe.sh NAME BATCH STEPS [extra lerobot-train args]
OUT=/home/saksham/experiments/libero_runs/probe
NAME=$1; BS=$2; STEPS=$3; shift 3
mkdir -p $OUT
cd /home/saksham/experiments/lerobot
# torchcodec needs FFmpeg shared libs: conda-forge ffmpeg 7 in the env (conda install -p .env -c conda-forge ffmpeg=7)
export LD_LIBRARY_PATH=$PWD/.env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $OUT/$NAME.gpu; sleep 2; done ) &
MON=$!
rm -rf $OUT/$NAME.train
/usr/bin/time -f "wall %e s, peak RAM %M KB" .env/bin/lerobot-train \
  --policy.path=lerobot/smolvla_libero --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=lerobot/libero --dataset.video_backend=torchcodec \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --batch_size=$BS --steps=$STEPS --log_freq=10 --save_checkpoint=false \
  --num_workers=${WORKERS:-4} --wandb.enable=false --output_dir=$OUT/$NAME.train "$@" \
  < /dev/null > $OUT/$NAME.log 2>&1
echo "exit $?" >> $OUT/$NAME.log
kill $MON
