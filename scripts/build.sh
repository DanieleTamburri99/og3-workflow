#!/bin/bash
# Build OpenGadget3 (P-Gadget3) on Leonardo with NVHPC 24.5 (OpenMP + OpenACC).
#
# Usage: build.sh <PMGRID> <heffte|fftw|scorep|hpctoolkit>
#   heffte      PM solver with GPU FFTs via HeFFTe (CUDA backend)
#   fftw        reference build with FFTW
#   scorep      Score-P instrumented build
#   hpctoolkit  build prepared for HPCToolkit profiling
#
# Build directory: build_<mode>_<PMGRID> (heffte/fftw) or build_<mode> (others).

GRID="$1"
MODE="$2"
VALID_MODES="heffte fftw scorep hpctoolkit"

if [ $# -lt 2 ] || [[ " ${VALID_MODES} " != *" ${MODE} "* ]]; then
    echo "Usage: $0 <grid dimension> <${VALID_MODES// /|}>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../config/env.sh"
if [ ! -f "$ENV_FILE" ]; then
    echo "Missing ${ENV_FILE}: copy config/env.example.sh to config/env.sh and edit it." >&2
    exit 1
fi
source "$ENV_FILE"
src_path="${SRC_PATH:?SRC_PATH is not set (see config/env.example.sh)}"

# ---------------------------------------------------------------- modules
module purge
module load cmake
module load nvhpc/24.5
module load fftw/3.3.10--hpcx-mpi--2.19--nvhpc--24.5
module load gsl/2.7.1--nvhpc--24.5
module load binutils/2.42-2wvnpkm
module load hdf5/1.14.3--hpcx-mpi--2.19--nvhpc--24.5

# CUDA shipped with the NVHPC module: <...>/24.5/compilers/bin/nvc -> <...>/24.5/cuda
NVC_BIN_DIR="$(dirname "$(which nvc)")"
export CUDA_ROOT="$(readlink -f "${NVC_BIN_DIR}/../../cuda")"
if [ ! -d "$CUDA_ROOT" ]; then
    echo "CUDA directory not found at ${CUDA_ROOT}: set CUDA_ROOT manually." >&2
    exit 1
fi
export CPLUS_INCLUDE_PATH=$CUDA_ROOT/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
export HDF5_ROOT=$HDF5_HOME
export LDFLAGS="-L$CUDA_ROOT/lib64 -lnvToolsExt"

# Fail fast from here on (kept after the module commands on purpose)
set -eo pipefail
trap 'echo "ERROR at line $LINENO of build.sh (see cmake_output.log / build_output.log in $PWD)" >&2' ERR

cd "$src_path"

# ---------------------------------------------------------------- configuration
case "$MODE" in
    heffte|fftw) BUILD_DIR="build_${MODE}_${GRID}" ;;
    *)           BUILD_DIR="build_${MODE}" ;;
esac

# Options shared by every build
COMMON=(-DINI_FILE=cmake/leonardo.cmake
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DOPENACC=ON
        -DNVTX=ON)

case "$MODE" in
    heffte)
        CMAKE_ARGS=("${COMMON[@]}" -DGPU_DEBUG=ON -DOPENMP=ON
                    -DUSE_HEFFTE=ON -DUSE_HEFFTE_CUDA=ON
                    -DCXX_CMAKE_COMPILER=nvc++)
        ;;
    fftw)
        CMAKE_ARGS=("${COMMON[@]}" -DOPENMP=ON
                    -DCXX_CMAKE_COMPILER=nvc++)
        ;;
    scorep)
        module load scorep/8.4--hpcx-mpi--2.19--nvhpc--24.5-cuda-12.2
        NVC_LIB="$(readlink -f "${NVC_BIN_DIR}/../lib/libnvc.so")"
        CMAKE_ARGS=("${COMMON[@]}"
                    -DCMAKE_C_COMPILER=scorep-mpicc
                    -DCMAKE_CXX_COMPILER=scorep-mpicxx
                    -DOpenMP_C_FLAGS="-mp" -DOpenMP_CXX_FLAGS="-mp"
                    -DOpenMP_C_LIB_NAMES="nvc" -DOpenMP_CXX_LIB_NAMES="nvc"
                    -DOpenMP_nvc_LIBRARY="$NVC_LIB"
                    -DUSE_HEFFTE=ON -DUSE_HEFFTE_CUDA=ON
                    -DCMAKE_VERBOSE_MAKEFILE=OFF)
        ;;
    hpctoolkit)
        CMAKE_ARGS=("${COMMON[@]}"
                    -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++
                    -DUSE_HEFFTE=ON -DUSE_HEFFTE_CUDA=ON
                    -DHPCTOOLKIT=ON
                    -DCMAKE_VERBOSE_MAKEFILE=OFF)
        ;;
esac

# Note: this overwrites Config.sh in the source tree (template + PMGRID)
cp template-Config.sh Config.sh
sed -i "s/^PMGRID=.*/PMGRID=${GRID}/" Config.sh

rm -rf "$BUILD_DIR"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# ---------------------------------------------------------------- build
cmake .. "${CMAKE_ARGS[@]}" > cmake_output.log 2>&1

EXIT_STATUS=0
make -j"${MAKE_JOBS:-$(nproc)}" > build_output.log 2>&1 || EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo "------------------------------------------------"
    echo "COMPILATION FAILED (Exit Code: $EXIT_STATUS)"
    echo "Error lines from build_output.log:"
    echo "------------------------------------------------"
    grep -i "error" build_output.log || true
    exit $EXIT_STATUS
fi

echo "------------------------------------------------"
echo "Compilation successful! (${src_path}/${BUILD_DIR})"
echo "------------------------------------------------"
tail -n 3 build_output.log
