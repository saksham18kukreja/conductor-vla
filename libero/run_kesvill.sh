#!/bin/sh
# Kesvill/smolvla_libero_90 (SmolVLA trained on LIBERO-90) on LIBERO-Long:
# full-instruction baseline, then M1 (each step alone, LIBERO-90 wording).
# n_action_steps=10 (checkpoint default 1): same LIBERO-90 sanity success
# (28/30 both ways) at 3x the speed.
R=/home/saksham/experiments/libero_runs/kesvill
cd /home/saksham/experiments/lerobot
export MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/.libero
ev() {  # out_name harness_config task_ids seed
  SUBGOAL_LOG=$R/$1_s$4.jsonl HARNESS_CONFIG="$2" \
  .env/bin/python ../franka_cube_lift/harness/eval_harness.py \
    --policy.path=Kesvill/smolvla_libero_90 --policy.n_action_steps=10 --env.type=libero --env.task=libero_10 \
    --env.task_ids="$3" --eval.batch_size=5 --eval.n_episodes=10 --env.max_parallel_tasks=1 --seed=$4 \
    --output_dir=$R/$1_s$4 --eval.use_async_envs=false --policy.device=cuda < /dev/null > $R/$1_s$4.log 2>&1
  echo "$1 seed $4 exit $?" >> $R/status.txt
}
for SEED in 1000 2000 3000; do
  ev subgoals '' '[0,1,2,3,4,5,6,7,8,9]' $SEED
  ev m1/step0_libero90 '{"steps": true, "only_step": 0, "wording": "libero90"}' '[0,1,2,3,4,6,7,8,9]' $SEED
  ev m1/step1_libero90 '{"steps": true, "only_step": 1, "wording": "libero90"}' '[0,1,2,3,4,6,7,8,9]' $SEED
done
echo ALLDONE >> $R/status.txt
