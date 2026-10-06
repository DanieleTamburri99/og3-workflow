#!/bin/bash
# Prepare a weak-scaling run directory, submit it to SLURM and follow the output.
#
# Usage:   submit.sh <GRID> <MODE>
# Example: submit.sh 1024 heffte
#          PROFILE=1 submit.sh 512 heffte      # run under Nsight Systems
#
# Weak scaling: the work per node is constant
#   512^3 -> 1 node, 1024^3 -> 8 nodes, 2048^3 -> 64 nodes (4 GPUs per node).
#
# PROFILE: 0 = no profiler, 1 = nsys, 2 = nsys stats, 3 = sanitizer,
#          4 = Score-P, 5 = HPCToolkit (see job.sh)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../config/env.sh"
if [ ! -f "$ENV_FILE" ]; then
    echo "Missing ${ENV_FILE}: copy config/env.example.sh to config/env.sh and edit it." >&2
    exit 1
fi
source "$ENV_FILE"

GRID="$1"
MODE="$2"
PROFILE="${PROFILE:-0}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <size> <mode>"
    exit 1
fi

# ---------------------------------------------------------------- run layout
declare -A NODES_FOR=( [32]=1 [512]=1 [1024]=8 [2048]=64 )   # 32 = quick test
NODES="${NODES_FOR[$GRID]:?Unknown size '$GRID' (valid: ${!NODES_FOR[*]})}"

RUN_DIR="${RUN_ROOT:?RUN_ROOT not set}/${GRID}/${MODE}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# ---------------------------------------------------------------- SLURM settings
JOB_NAME="BOG"
PARTITION="${PARTITION:-boost_usr_prod}"
QOS="normal"
NTASKS_PER_NODE=4      # one MPI task per GPU
CPUS_PER_TASK=8        # OpenMP threads per MPI task
TIME="02:00:00"
ACCOUNT="${SLURM_ACCOUNT:?SLURM_ACCOUNT not set (see config/env.example.sh)}"

# ---------------------------------------------------------------- parameter file
IC_PATH="${IC_ROOT:?IC_ROOT not set}/${GRID}/ics_${GRID}"
paramfile="${IC_ROOT}/${GRID}/param_${GRID}.par"
if [ ! -f "$paramfile" ]; then
    echo "Parameter file not found: $paramfile" >&2
    exit 1
fi

# set_param KEY VALUE FILE: replace the value if KEY exists, append it otherwise
set_param() {
    if grep -q "^$1[[:space:]]" "$3"; then
        sed -i "s#^$1[[:space:]].*#$1 $2#" "$3"
    else
        echo "$1 $2" >> "$3"
    fi
}

cp "$paramfile" param.par
set_param StopAfterNSteps       100        param.par
set_param TimerReportLevel      4          param.par
set_param InitCondFile          "$IC_PATH" param.par
set_param CpuTimeBetRestartFile 7200       param.par
set_param ICFormat              1          param.par

# Scale factors of the snapshots (z = 12, 3, 2, 0.5)
echo -e "0.076923\n0.25\n0.333333\n0.666667" > outputs

# ---------------------------------------------------------------- submit
JOB_ID=$(sbatch --parsable \
    --nodes="${NODES}" \
    --ntasks-per-node="${NTASKS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gres=gpu:4 \
    --time="${TIME}" \
    --partition="${PARTITION}" \
    --qos="${QOS}" \
    --output="${RUN_DIR}/%j.out" \
    --account="${ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --exclusive \
    --mem=0 \
    "${SCRIPT_DIR}/job.sh" --profile "${PROFILE}" --grid "${GRID}" --mode "${MODE}")
JOB_ID="${JOB_ID%%;*}"     # --parsable may append ";cluster"
OUT_FILE="${RUN_DIR}/${JOB_ID}.out"

echo "Job ${JOB_ID} submitted (${NODES} node(s)). Waiting for ${OUT_FILE}"
echo "Ctrl-C stops the tail only: the job stays in the queue."
tail -F "$OUT_FILE"     # -F waits for the file to appear (job may be pending)
