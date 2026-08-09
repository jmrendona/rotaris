#!/bin/bash

# Runs convert.py (SNCReader / pf2ens .snc -> HDF5 conversion) as a single-node
# batch job. Nothing in convert.py is parallelized (see README.md), so this
# only ever requests one node / one task - raise --cpus-per-task or --mem
# below if a given case needs more, but don't add --ntasks/--nodes without
# also parallelizing convert.py itself.
#
# Usage:
#   sbatch run_conversion.sh forces   <snc_path> <output.h5> [--surface-split] [--face-name NAME]
#   sbatch run_conversion.sh pressure <snc_path> <output.h5> --first N --last M [--surface-split] [--nc-stats FILE] [--reference-frame N] [--work-dir DIR]
#
# Everything after the job script path is forwarded to convert.py as-is.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00               # ADJUST ME - depends on file size / frame count
#SBATCH --account=rrg-moreaust-ac
#SBATCH --job-name=rotaris-convert
#SBATCH --output=%x_%j_out.txt
#SBATCH --error=%x_%j_err.txt
#SBATCH --mail-user=jmrendon@usherbrooke.ca   # ADJUST ME if needed
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# =-=-=-=-=-=-=-=-=-=-=-=-=-
# POWERFLOW VERSION (for pf2ens - only actually needed by `pressure`, but
# harmless to load for `forces` too)
use_pf_version="6-2025-R3"
source /project/rrg-moreaust-ac/Env/powerflow_env.sh $use_pf_version

# =-=-=-=-=-=-=-=-=-=-=-=-=-
# PYTHON ENVIRONMENT
#
# NOTE: Alliance/Compute Canada compute nodes have NO internet access, so the
# block below will fail if this is the very first run and the venv doesn't
# exist yet. Build it once from a LOGIN node before your first `sbatch`
# submission:
#
#   module load StdEnv/2023 python/3.11 scipy-stack/2023b vtk
#   virtualenv --no-download --system-site-packages ~/rotaris-venv
#   source ~/rotaris-venv/bin/activate
#   pip install --no-index --upgrade pip
#   pip install pyvista
#
# `virtualenv --no-download` (not `python -m venv`) matters here: Alliance's
# python module ships without ensurepip bundled, so `python -m venv` fails
# trying to fetch pip. `virtualenv` knows to skip that and pulls pip from
# Alliance's own local wheel mirror instead (`--no-index`). pyvista turned
# out to already be in that same local mirror (cvmfs wheelhouse), so this
# whole block actually runs fine without real internet either way - it's
# still meant to run once from a login node, though, since compute nodes
# aren't guaranteed the same wheelhouse access.
#
# After that, this script just reuses the venv - no internet needed on the
# compute node.

module load StdEnv/2023 python/3.11 scipy-stack/2023b vtk

VENV_DIR="$HOME/rotaris-venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "No venv found at $VENV_DIR - build it from a login node first, see the note above." >&2
    exit 1
else
    source "$VENV_DIR/bin/activate"
fi

# =-=-=-=-=-=-=-=-=-=-=-=-=-
# RUN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/convert.py" "$@"
