#!/usr/bin/env bash

ROOT_DIR=$1
map=$2
runs=$3
samples=$4
data_format=$5

# 

COMPLETE_PATH=${ROOT_DIR}/${map}/NN/${data_format}
# echo ${COMPLETE_PATH}
COMPLETE_PATH="$(realpath "$COMPLETE_PATH")"
# echo ${COMPLETE_PATH}
echo "Root path: $COMPLETE_PATH"

# Opzioni opzionali controllate via env
EXTRA_ARGS=()

# count=0
# echo $COMPLETE_PATH
# while IFS= read -r -d '' plan_path; do
#   ((count++)) || true
#   plan_dir="$(dirname "$plan_path")"
#   leaf_name="$(basename "$plan_dir")"
#   out_json="$plan_dir/${leaf_name}_times.json"

#   echo "[$count] Profiling: $plan_path"
#   echo "     -> ${out_json}"

#   trtexec \
#     --iterations="${runs}" \
#     --loadEngine="${plan_path}" \
#     --dumpProfile \
#     --exportTimes="${plan_path%%.*}_times.json" \
#     --profilingVerbosity=detailed \
#     --separateProfileRun \
#     "${EXTRA_ARGS[@]}"

# done < <(find "$COMPLETE_PATH" -type f -name '*.plan' -print0)

# echo "Completato. Processati $count engine .plan"

python trt_benchmarking.py --root ${COMPLETE_PATH} --runs ${runs} --samples ${samples}

python compare.py ${COMPLETE_PATH} -o out_report/${map}/