#!/bin/sh
# Harness M2: steps in sequence (LIBERO-90 wording, predicate verifier), no
# recovery, Kesvill/smolvla_libero_90, the 6 tasks whose steps are all in
# LIBERO-90. Batch size 1 (the VLA's action queue is shared across a batch).
R=/home/saksham/experiments/libero_runs/kesvill/m2
cd /home/saksham/experiments/lerobot
export MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/.libero
for SEED in 1000 2000 3000; do
  SUBGOAL_LOG=$R/subgoals_s$SEED.jsonl HARNESS_LOG=$R/harness_s$SEED.jsonl \
  HARNESS_CONFIG='{"steps": true, "wording": "libero90"}' \
  .env/bin/python ../franka_cube_lift/harness/eval_harness.py \
    --policy.path=Kesvill/smolvla_libero_90 --policy.n_action_steps=10 --env.type=libero --env.task=libero_10 \
    --env.task_ids='[0,2,3,4,6,7]' --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 --seed=$SEED \
    --output_dir=$R/s$SEED --eval.use_async_envs=false --policy.device=cuda < /dev/null > $R/s$SEED.log 2>&1
  echo "seed $SEED exit $?" >> $R/status.txt
done
echo ALLDONE >> $R/status.txt
