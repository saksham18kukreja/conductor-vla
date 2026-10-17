#!/bin/sh
# Quick check of a fine-tuned checkpoint (seed 1000, 10 episodes per task):
#  steps alone (harness M1, conductor wording): the 3 steps LIBERO-90 lacks
#    (task 1 cream cheese, task 8 left moka pot, task 9 mug into microwave)
#    and 2 it has (task 2 moka pot on stove, task 4 white mug on left plate)
#  full instruction alone: tasks 2 and 4 (does it still do whole tasks?)
#   usage: eval_compare.sh MODEL_DIR NAME
MODEL=$1; NAME=$2
R=/home/saksham/experiments/libero_runs/finetune/eval_$NAME
mkdir -p $R
cd /home/saksham/experiments/lerobot
export MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/.libero
ev() {  # out harness_config task_ids
  SUBGOAL_LOG=$R/$1.jsonl HARNESS_CONFIG="$2" .env/bin/python ../franka_cube_lift/harness/eval_harness.py \
    --policy.path=$MODEL --env.type=libero --env.task=libero_10 --env.task_ids="$3" \
    --eval.batch_size=5 --eval.n_episodes=10 --env.max_parallel_tasks=1 --seed=1000 \
    --output_dir=$R/$1 --eval.use_async_envs=false --policy.device=cuda \
    '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
    --policy.empty_cameras=1 < /dev/null > $R/$1.log 2>&1
  echo "$1 exit $?" >> $R/status.txt
}
ev step0 '{"steps": true, "only_step": 0, "wording": "conductor"}' '[1,8,9,4]'
ev step1 '{"steps": true, "only_step": 1, "wording": "conductor"}' '[2]'
ev full '' '[2,4]'
echo DONE >> $R/status.txt
