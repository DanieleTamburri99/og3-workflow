# Copy this file to config/env.sh and edit it. config/env.sh is git-ignored.
# NOTE: every variable must be exported, because sbatch propagates the
# environment to job.sh.

export SRC_PATH="$HOME/OpenGadget3"                   # OpenGadget3 sources (not part of this repo)
export SLURM_ACCOUNT="your_project_account"           # SLURM account / budget
export PARTITION="boost_usr_prod"                     # Leonardo Booster partition
export RUN_ROOT="$SCRATCH/OpenGadget3/weak_scaling"   # where runs are created
export IC_ROOT="$SCRATCH/ICs"                         # <IC_ROOT>/<GRID>/{ics_<GRID>,param_<GRID>.par}
export HPCTOOLKIT_DIR=""                              # only needed for PROFILE=5
