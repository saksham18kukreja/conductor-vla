#!/bin/sh
# LIBERO eval on this laptop: EGL is broken (driver/library mismatch), so
# render through the X display with GLX; robosuite only skips EGL for glx/osmesa.
# usage: run_libero_eval.sh POLICY SUITE SEED OUTDIR [extra lerobot-eval args]
cd "$(dirname "$0")"
export MUJOCO_GL=glx DISPLAY=${DISPLAY:-:0} LIBERO_CONFIG_PATH=$PWD/.libero
POLICY=$1; SUITE=$2; SEED=$3; OUT=$4; shift 4
exec .env/bin/lerobot-eval --policy.path="$POLICY" --env.type=libero --env.task="$SUITE" \
  --eval.batch_size=10 --eval.n_episodes=10 --env.max_parallel_tasks=1 --seed="$SEED" \
  --output_dir="$OUT" "$@" < /dev/null
