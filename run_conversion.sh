#!/bin/bash

# Runs convert.py (.snc / .fnc -> HDF5 conversion) as a single-node batch
# job. Nothing in convert.py is parallelized (see README.md), so this only
# ever requests one node / one task - raise --cpus-per-task or --mem below
# if a given case needs more, but don't add --ntasks/--nodes without also
# parallelizing convert.py itself.
#
# Usage:
#   sbatch run_conversion.sh forces               <snc_path> <output.h5> [--surface-split] [--face-name NAME]
#   sbatch run_conversion.sh pressure             <snc_path> <output.h5> --first N --last M [--surface-split] [--nc-stats FILE] [--reference-frame N] [--work-dir DIR]
#   sbatch run_conversion.sh fnc-meridional       <fnc_path> <output.h5>  --angle DEG --variables v1,v2 --first N --last M [--freeze-mask-variable v1] [--plot out.png --plot-variable v1]
#   sbatch run_conversion.sh fnc-meridional-sweep <fnc_path> <output_dir> --angle-start A0 --angle-end A1 --angle-step DA --variables v1,v2 --first N --last M [--freeze-mask-variable v1]
#   sbatch run_conversion.sh fnc-iso-radius       <fnc_path> <output.h5>  --radius R --variables v1,v2 --first N --last M
#   sbatch run_conversion.sh fnc-points           <fnc_path> <output.h5>  --points-file points.txt --variables v1,v2 --first N --last M
#
# Everything after the job script path is forwarded to convert.py as-is.
#
# fnc-meridional/fnc-iso-radius/fnc-points each write ONE HDF5 file for the
# ONE plane/surface/point-cloud you asked for (all requested frames and
# variables together inside it). For multiple angles in ONE command/job, use
# fnc-meridional-sweep instead - one HDF5 file per angle, written into
# <output_dir>, sharing a single nc-stats.ri call across every angle:
#
#   sbatch run_conversion.sh fnc-meridional-sweep case.fnc /path/to/output_dir \
#       --angle-start 0 --angle-end 170 --angle-step 10 \
#       --variables vmag --first 0 --last 96 --freeze-mask-variable vmag
#
# That's 18 angles x 97 frames = 1746 pf2ens calls at ~3 min each - budget
# --time accordingly (well beyond the 02:00:00 default below), or split the
# sweep across multiple sbatch submissions (e.g. one per angle, the old way -
# more scheduling overhead but each job is short and they run in parallel):
#
#   for angle in 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170; do
#       sbatch run_conversion.sh fnc-meridional case.fnc "plane_${angle}deg.h5" \
#           --angle "$angle" --variables vmag --first 0 --last 96
#   done

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00               # ADJUST ME - depends on file size / frame count
#SBATCH --account=rrg-moreaust-ac
#SBATCH --job-name=rotaris-convert
#SBATCH --output=%x_%j_out.txt
#SBATCH --error=%x_%j_err.txt
#SBATCH --mail-user=jmrendon@usherbrooke.ca   # ADJUST ME if needed
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# =-=-=-=-=-=-=-=-=-=-=-=-=-
# POWERFLOW VERSION (for pf2ens/exaritool - needed by `pressure` and every
# `fnc-*` subcommand)
#
# MUST be 6-2024-R1: `pf2ens` does not exist under 6-2025-R3's install at
# all (confirmed empirically - only exaritool and a few other exa* tools
# are there), only under 6-2024-R1.
use_pf_version="6-2024-R1"
source /project/rrg-moreaust-ac/Env/powerflow_env.sh $use_pf_version

# =-=-=-=-=-=-=-=-=-=-=-=-=-
# PYTHON ENVIRONMENT
#
# gcc/12.3 + vtk/9.4.2 are required for `import pyvista` to actually work
# (used by ensight_to_h5.py for `pressure`, and by fnc_plane.py for every
# `fnc-*` subcommand) - `pip install pyvista` alone is NOT enough, `vtk`
# itself has no installable wheel (checked both Compute Canada's wheelhouse
# and public PyPI); it must come from this module. python/3.11 MUST load
# before vtk/9.4.2 (Lmod's extension mechanism binds the Python-version-
# specific bindings at that point, not after).
#
# NOTE: Alliance/Compute Canada compute nodes usually have NO internet
# access, so `pip install` below will fail if this is the very first run
# and the venv doesn't exist yet. Build it once from a LOGIN node before
# your first `sbatch` submission:
#
#   module load StdEnv/2023 gcc/12.3 python/3.11 vtk/9.4.2 scipy-stack/2023b
#   python -m venv --system-site-packages ~/rotaris-venv
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
# After that, this script just reuses it - no internet needed on the compute
# node. IMPORTANT: never pipe a `module load` line through anything (e.g.
# `| tail`) - that runs it in a subshell and silently drops every
# environment change it makes, so nothing after it (pyvista, vtk, even
# plain numpy) will actually be on PYTHONPATH.

module load StdEnv/2023 gcc/12.3 python/3.11 vtk/9.4.2 scipy-stack/2023b
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

VENV_DIR="$HOME/rotaris-venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "No venv found at $VENV_DIR - build it from a login node first, see the note above." >&2
    exit 1
else
    source "$VENV_DIR/bin/activate"
fi

# =-=-=-=-=-=-=-=-=-=-=-=-=-
# RUN
#
# NOT ${BASH_SOURCE[0]}'s directory: `sbatch` COPIES this script to a
# per-job spool directory on the compute node
# (/var/spool/slurm/slurmd/job<ID>/) and runs THAT copy - so
# ${BASH_SOURCE[0]} resolves to the spool path, not wherever this script
# actually lives, and `convert.py` isn't there (only this one file got
# copied). ROTARIS_DIR is a fixed, known install location instead - same
# pattern as VENV_DIR above. Override it if rotaris isn't at $HOME/rotaris:
#   ROTARIS_DIR=/path/to/rotaris sbatch run_conversion.sh ...

ROTARIS_DIR="${ROTARIS_DIR:-$HOME/rotaris}"

if [ ! -f "$ROTARIS_DIR/convert.py" ]; then
    echo "No convert.py found at $ROTARIS_DIR - set ROTARIS_DIR to wherever rotaris actually lives." >&2
    exit 1
fi

python "$ROTARIS_DIR/convert.py" "$@"
