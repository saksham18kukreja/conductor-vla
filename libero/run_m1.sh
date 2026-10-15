#!/bin/sh
# Harness M1: each plan step alone from the task's initial state, full 520-step budget.
R=/home/saksham/experiments/libero_runs/m1
cd /home/saksham/experiments/lerobot
export MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/.libero
run() {  # name step wording task_ids seed
  SUBGOAL_LOG=$R/$1_s$5.jsonl HARNESS_CONFIG="{\"steps\": true, \"only_step\": $2, \"wording\": \"$3\"}" \
  .env/bin/python ../franka_cube_lift/harness/eval_harness.py \
    --policy.path=lerobot/smolvla_libero --env.type=libero --env.task=libero_10 --env.task_ids="$4" \
    --eval.batch_size=5 --eval.n_episodes=10 --env.max_parallel_tasks=1 --seed=$5 \
    --output_dir=$R/$1_s$5 --eval.use_async_envs=false --policy.device=cuda \
    '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
    --policy.empty_cameras=1 < /dev/null > $R/$1_s$5.log 2>&1
  echo "$1 seed $5 exit $?" >> $R/status.txt
}
for SEED in 1000 2000 3000; do
  run step0_trained 0 trained '[0,1,2,3,4,6,7,8,9]' $SEED
  run step1_trained 1 trained '[0,1,2,3,4,6,7,8,9]' $SEED
  run step0_split   0 split   '[0,1,7]' $SEED
  run step1_split   1 split   '[0,1,2,3,7,9]' $SEED
done
echo ALLDONE >> $R/status.txt
