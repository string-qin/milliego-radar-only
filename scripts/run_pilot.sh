cd /home/metaiot_guest/Workspace/qinzx/milliEgo || exit 1
source ./env.sh
export CUDA_VISIBLE_DEVICES=3
timeout 1200 .venv/bin/python -u scripts/train_single_sequence.py --epochs 200 --overfit-steps 400 --learning-rates 1e-5,1e-4 --output runs/pilot_lab0_gap5_v1
rc=$?
printf '%s\n' "$rc" > reports/single_sequence_train.exit
exit "$rc"
