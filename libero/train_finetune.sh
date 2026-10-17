#!/bin/sh
# Fine-tune lerobot/smolvla_libero on the Conductor dataset (libero/build_dataset.py)
# on this laptop (RTX 4060, 8 GB): batch 16 + torch.compile (cost probe:
# ~1.9 steps/s, < 5 GB). Saves the final model to OUT/NAME/checkpoints/last.
#   usage: train_finetune.sh NAME STEPS lora|expert [extra lerobot-train args]
OUT=/home/saksham/experiments/libero_runs/finetune
DATA=/home/saksham/experiments/libero_runs/libero_conductor_dataset
NAME=$1; STEPS=$2; MODE=$3; shift 3
WARMUP=${WARMUP:-$((STEPS / 10))}
SAVE_FREQ=${SAVE_FREQ:-$STEPS}
case $MODE in
  # LoRA on the language model's and expert's q/v + action/state projections; lr x10 (LeRobot PEFT guide)
  lora) M="--peft.method_type=LORA --peft.r=32 --peft.lora_alpha=32 --policy.optimizer_lr=1e-3 --policy.scheduler_decay_lr=1e-4"
        T='--peft.target_modules=(.*\.text_model\..*\.(q_proj|v_proj)|.*\.lm_expert\..*\.(q_proj|v_proj)|.*\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))' ;;
  # SmolVLA's default: train the action expert (+ state projection), vision and language frozen
  expert) M="--policy.train_expert_only=true --policy.freeze_vision_encoder=true --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=2.5e-6"; T="" ;;
  *) echo "mode must be lora or expert"; exit 2 ;;
esac
mkdir -p $OUT
cd /home/saksham/experiments/lerobot
export LD_LIBRARY_PATH=$PWD/.env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
rm -rf "$OUT/$NAME"
/usr/bin/time -f "wall %e s, peak RAM %M KB" .env/bin/lerobot-train \
  --policy.path=lerobot/smolvla_libero --policy.push_to_hub=false --policy.device=cuda \
  --policy.compile_model=true --policy.compile_mode=default \
  --policy.scheduler_warmup_steps=$WARMUP --policy.scheduler_decay_steps=$STEPS \
  --dataset.repo_id=saksham18kukreja/libero_conductor --dataset.root=$DATA --dataset.video_backend=torchcodec \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --batch_size=16 --steps=$STEPS --log_freq=100 --save_checkpoint=true --save_freq=$SAVE_FREQ \
  --num_workers=4 --wandb.enable=false --seed=1000 --output_dir=$OUT/$NAME $M ${T:+"$T"} "$@" \
  < /dev/null > $OUT/$NAME.log 2>&1
echo "TRAIN_EXIT $?" >> $OUT/$NAME.log
