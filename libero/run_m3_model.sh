#!/bin/sh
# Conductor (harness M3: steps, start-state check + recovery, retry) around a
# fine-tuned SmolVLA, all 10 LIBERO-Long tasks, conductor wording, batch 1.
#   usage: run_m3_model.sh MODEL_DIR OUT_DIR "SEEDS"
MODEL=$1; R=$2; SEEDS=${3:-1000}
mkdir -p $R
cd /home/saksham/experiments/lerobot
export MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/.libero
for SEED in $SEEDS; do
  SUBGOAL_LOG=$R/subgoals_s$SEED.jsonl HARNESS_LOG=$R/harness_s$SEED.jsonl \
  HARNESS_CONFIG=${HARNESS_CONFIG:-'{"steps": true, "wording": "conductor", "recover": true, "retry": true}'} \
  .env/bin/python ../franka_cube_lift/harness/eval_harness.py \
    --policy.path=$MODEL --env.type=libero --env.task=libero_10 --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
    --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 --seed=$SEED \
    --output_dir=$R/s$SEED --eval.use_async_envs=false --policy.device=cuda \
    '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
    --policy.empty_cameras=1 < /dev/null > $R/s$SEED.log 2>&1
  echo "seed $SEED exit $?" >> $R/status.txt
done
echo ALLDONE >> $R/status.txt
