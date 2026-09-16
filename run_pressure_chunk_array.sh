#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00               # overridden by submit_pressure_chunks.sh's --time
#SBATCH --account=rrg-moreaust-ac
#SBATCH --job-name=pchunk
#SBATCH --output=pchunk_%A_%a_out.txt
#SBATCH --error=pchunk_%A_%a_err.txt

# One SLURM array TASK per frame chunk - always submitted via
# submit_pressure_chunks.sh (which sets --array=0-N%K), never by hand.
# K throttles how many chunks SLURM runs CONCURRENTLY - see that
# script's own header for why this exists: submitting all chunks as
# independent jobs at once (the previous version of this pipeline) let
# too many pf2ens processes compete for this account's shared, LIMITED
# PowerFLOW license pool (exasignalprocessingjob) at the same instant -
# confirmed on a real run: several chunks died individually ~2 minutes
# after their own start, each killed by an automated system process
# (UID 0), consistent with a license-acquisition grace period expiring.
# A throttled array keeps concurrent license demand bounded instead.
#
# Positional args (same for every task in the array - only
# $SLURM_ARRAY_TASK_ID differs task to task, which is what picks out
# THIS task's own frame chunk):
#   $1 = snc_path
#   $2 = output_dir
#   $3 = first_frame (of the WHOLE requested range, not this chunk)
#   $4 = chunk_size
#   $5 = last_frame (of the whole range - this chunk's own --last is
#        clamped to it, so the final chunk can be shorter than chunk_size)
#   $6.. = extra convert.py args (e.g. --surface-split), forwarded as-is

set -euo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "SLURM_ARRAY_TASK_ID is not set - this script is meant to run as a SLURM job " \
         "array task (via submit_pressure_chunks.sh --array=...), not standalone." >&2
    exit 1
fi

SNC_PATH="$1"; OUTPUT_DIR="$2"; FIRST_FRAME="$3"; CHUNK_SIZE="$4"; LAST_FRAME="$5"
shift 5

chunk_start=$(( FIRST_FRAME + SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
chunk_end=$(( chunk_start + CHUNK_SIZE - 1 ))
if [ "$chunk_end" -gt "$LAST_FRAME" ]; then
    chunk_end=$LAST_FRAME
fi

output_file="$OUTPUT_DIR/pressure_frames_$(printf '%04d' "$chunk_start")_$(printf '%04d' "$chunk_end").h5"

# Same ROTARIS_DIR convention as run_conversion.sh itself (not this
# script's own path - sbatch copies it to a per-job spool dir too).
# run_conversion.sh's OWN #SBATCH directives are just comments at this
# point (we're calling it as a plain subprocess from inside an already-
# allocated job, not submitting it fresh) - only its env-setup + `python
# convert.py "$@"` body actually runs, which is exactly what's wanted here.
ROTARIS_DIR="${ROTARIS_DIR:-$HOME/rotaris}"

exec "$ROTARIS_DIR/run_conversion.sh" pressure "$SNC_PATH" "$output_file" \
    --first "$chunk_start" --last "$chunk_end" "$@"
