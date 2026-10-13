#!/bin/sh
# SmolVLA (lerobot/smolvla_libero) on LIBERO-Long, 3 seeds, settings from LeRobot CI.
R=/home/saksham/experiments/libero_runs
for SEED in 1000 2000 3000; do
  /home/saksham/experiments/lerobot/run_libero_eval.sh lerobot/smolvla_libero libero_10 $SEED $R/smolvla_libero10_s$SEED \
    --eval.use_async_envs=false --eval.batch_size=5 --eval.n_episodes=10 --policy.device=cuda \
    '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
    --policy.empty_cameras=1 > $R/smolvla_libero10_s$SEED.log 2>&1
  echo "seed $SEED exit $?" >> $R/baseline_status.txt
done
echo ALLDONE >> $R/baseline_status.txt
