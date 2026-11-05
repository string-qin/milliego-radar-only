. ./env.sh
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4
for spec in 1:4 2:4 5:4 3:2; do
 w=${spec%:*}
 k=${spec#*:}
 name=window${w}_scale${k}
 .venv/bin/python -u scripts/prepare_projection_sweep.py "$w" "$k" > "reports/${name}.log" 2>&1 &&
 .venv/bin/python -u "scripts/validate_${name}.py" >> "reports/${name}.log" 2>&1
 echo $? > "reports/${name}.exit"
done
