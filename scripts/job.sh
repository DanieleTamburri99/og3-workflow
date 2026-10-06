#!/bin/bash
# Executed inside the SLURM allocation (submitted by submit.sh).
#
# Usage (via sbatch): job.sh --mode <heffte|fftw> --grid <N> [--profile <0-5>]
#
# PROFILE:
#   0  plain run
#   1  Nsight Systems (one report per MPI rank)
#   2  Nsight Systems with --stats
#   3  compute-sanitizer (memcheck)
#   4  Score-P (needs the build made with: build.sh <N> scorep)
#   5  HPCToolkit (needs the build made with: build.sh <N> hpctoolkit)
#
# The environment exported by config/env.sh is propagated by sbatch.

set -eo pipefail

PROFILE=0
src_path="${SRC_PATH:?SRC_PATH not set: source config/env.sh before calling sbatch}"
init_mod=0   # 2nd argument of P-Gadget3 (RestartFlag): 0 = start from ICs

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)     MODE="$2";    shift 2 ;;
    --grid)     GRID="$2";    shift 2 ;;
    --profile)  PROFILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
: "${MODE:?missing --mode}"
: "${GRID:?missing --grid}"

# ---------------------------------------------------------------- threads
num_tasks=${SLURM_NTASKS}            # number of MPI tasks
num_threads=${SLURM_CPUS_PER_TASK}   # number of OpenMP threads per MPI process
export OMP_NUM_THREADS=$num_threads
export OMP_PROC_BIND=close           # binding of OpenMP threads
export OMP_PLACES=cores              # places of OpenMP threads

# ---------------------------------------------------------------- modules (Leonardo)
module load nvhpc/24.5
module load fftw/3.3.10--hpcx-mpi--2.19--nvhpc--24.5
module load gsl/2.7.1--nvhpc--24.5
module load binutils/2.42-2wvnpkm
module load hdf5/1.14.3--hpcx-mpi--2.19--nvhpc--24.5

export PGI_ACC_MEM_MANAGE=0
export UCX_TLS=rc,cuda_copy,gdr_copy

# ---------------------------------------------------------------- launch
BIN="${src_path}/build_${MODE}_${GRID}/P-Gadget3"
MPI_ARGS=(-np "$num_tasks"
          --map-by "ppr:${SLURM_NTASKS_PER_NODE}:node:pe=${OMP_NUM_THREADS}")
WRAP=()   # optional profiler/debugger placed in front of the executable

case "$PROFILE" in
  0) ;;

  1) # Nsight Systems
     mkdir -p profiles
     WRAP=(nsys profile --trace=nvtx,cuda,openacc,mpi,osrt,ucx
                        --cuda-memory-usage=true
                        --cudabacktrace=all
                        --mpi-impl=openmpi
                        --force-overwrite=true
                        --output="profiles/report_${MODE}_N${GRID}_rank_%q{OMPI_COMM_WORLD_RANK}")
     ;;

  2) # Nsight Systems with summary statistics
     mkdir -p profiles
     WRAP=(nsys profile --trace=nvtx,cuda,openacc,mpi,osrt
                        --mpi-impl=openmpi
                        --cuda-memory-usage=true
                        --stats=true
                        --force-overwrite=true
                        --output="profiles/report_stats_${MODE}_N${GRID}_rank_%q{OMPI_COMM_WORLD_RANK}")
     ;;

  3) # compute-sanitizer
     export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
     WRAP=("$(which compute-sanitizer)" --tool memcheck
                                        --launch-timeout 180
                                        --target-processes all
                                        --log-file sanitizer_rank_%p.log)
     ;;

  4) # Score-P
     BIN="${src_path}/build_scorep/P-Gadget3"
     export SCOREP_EXPERIMENT_DIRECTORY="scorep_results"
     mkdir -p scorep_results
     export SCOREP_ENABLE_PROFILING=true
     export SCOREP_ENABLE_TRACING=true
     export SCOREP_TOTAL_MEMORY=4G
     export SCOREP_MPI_ENABLE_GROUPS=ENV,COLL,P2P
     export SCOREP_ENABLE_UNWINDING=true
     export SCOREP_VERBOSE=true
     unset SCOREP_FILTERING_FILE     # no filtering: instrument everything
     echo "=========================================="
     echo "Job ID: $SLURM_JOB_ID"
     echo "Running on $num_tasks MPI tasks"
     echo "OpenMP threads per task: $num_threads"
     echo "Score-P results will be in: $SCOREP_EXPERIMENT_DIRECTORY"
     echo "=========================================="
     ;;

  5) # HPCToolkit
     BIN="${src_path}/build_hpctoolkit/P-Gadget3"
     : "${HPCTOOLKIT_DIR:?set HPCTOOLKIT_DIR in config/env.sh}"
     export PATH="${HPCTOOLKIT_DIR}/bin:$PATH"
     WRAP=(hpcrun -e CPUTIME -e gpu=nvidia -tt)
     ;;

  *) echo "Unknown PROFILE value: $PROFILE (expected 0-5)"; exit 1 ;;
esac

mpirun "${MPI_ARGS[@]}" "${WRAP[@]}" "$BIN" param.par "$init_mod"

# HPCToolkit post-processing
if [ "$PROFILE" -eq 5 ]; then
    echo -e "\nHPCTOOLKIT: Recovering program structure"
    hpcstruct "hpctoolkit-P-Gadget3-measurements-${SLURM_JOB_ID}"
    echo -e "\nHPCTOOLKIT: Attributing performance to code"
    hpcprof   "hpctoolkit-P-Gadget3-measurements-${SLURM_JOB_ID}"
fi
