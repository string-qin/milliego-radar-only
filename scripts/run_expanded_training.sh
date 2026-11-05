cd /home/metaiot_guest/Workspace/qinzx/milliEgo || exit 1
source ./env.sh
for attempt in $(seq 1 90); do
    if test -s data/coloradar_lab_expanded_v1/manifest.json; then break; fi
    sleep 10
done
if ! test -s data/coloradar_lab_expanded_v1/manifest.json; then
    echo "Data preparation did not complete; training not started"
    echo 2 > reports/expanded_training.exit
    exit 2
fi
export CUDA_VISIBLE_DEVICES=3
timeout 1800 .venv/bin/python -u scripts/validate_expanded_training.py
rc=$?
if test "$rc" -eq 0; then
    .venv/bin/python scripts/report_expanded_training.py > reports/expanded_training_summary.json
    rc=$?
fi
printf '%s\n' "$rc" > reports/expanded_training.exit
exit "$rc"
