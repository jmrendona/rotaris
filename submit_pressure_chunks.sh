#!/bin/bash

# Submit one `run_conversion.sh pressure` job PER CHUNK of frames, all
# submitted at once with NO --dependency between them (parallel, not
# chained) - see the wall-clock reasoning below for why.
#
# WHY THIS EXISTS: convert_snc_to_h5() calls pf2ens once per frame, and
# each call is a fresh, independent read of the WHOLE source .snc -
# confirmed at ~16.2 min/frame on a real case
# (/scratch/jmrendon/Rotor-alone/6e-5_6000rpm_HF/SMF_fwh_rotor.snc,
# ~858 MB/frame, 773 frames requested) - a single job covering that many
# frames runs well past this cluster's `compute` partition time-limit
# ceiling (24h - see `sinfo -p compute`; the job that hit this only got
# through 37 frames in its 10h limit before being killed). Splitting
# into N smaller chunk jobs, run IN PARALLEL, finishes the whole range in
# roughly the time of ONE chunk instead of the sum of all of them -
# chosen over a sequential (--dependency=afterany) chain after checking
# this account's PowerFLOW license usage (`exasignalprocessingjob`: 126
# total, 80 in use at check time - enough headroom for a few dozen
# concurrent chunk jobs). If license/queue pressure becomes a real
# problem, switch to a sequential chain instead (afterany, not afterok -
# so one failed chunk, e.g. its own time-limit, doesn't stall every
# chunk after it waiting on a dependency that can never succeed).
#
# Writes ONE output .h5 PER CHUNK
# (<output_dir>/pressure_frames_<first>_<last>.h5) - convert_snc_to_h5()
# always creates a fresh file (h5py.File(path, 'w')), it can't append to
# an existing one, so a single shared output file across chunks was
# never an option. Merging the chunk files into one afterward is a
# SEPARATE step, not done here.
#
# Usage:
#   ./submit_pressure_chunks.sh <snc_path> <output_dir> <first_frame> <last_frame> <chunk_size> <time_per_chunk> [extra convert.py args...]
#
# Example (this session's actual case - 773 frames, 30/chunk, 12h/chunk
# budget against an expected ~8.1h/chunk at this file's own measured rate):
#   ./submit_pressure_chunks.sh \
#       /scratch/jmrendon/Rotor-alone/6e-5_6000rpm_HF/SMF_fwh_rotor.snc \
#       /scratch/jmrendon/Rotor-alone/6e-5_6000rpm_HF/pressure_chunks \
#       0 772 30 12:00:00 --surface-split

set -euo pipefail

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <snc_path> <output_dir> <first_frame> <last_frame> <chunk_size> <time_per_chunk> [extra convert.py args...]" >&2
    exit 1
fi

SNC_PATH="$1"; OUTPUT_DIR="$2"; FIRST="$3"; LAST="$4"; CHUNK_SIZE="$5"; TIME_PER_CHUNK="$6"
shift 6
EXTRA_ARGS=("$@")

mkdir -p "$OUTPUT_DIR"

# NOT ${BASH_SOURCE[0]}'s directory alone - this script (unlike
# run_conversion.sh) is run directly from a login node, not copied to a
# compute-node spool dir by sbatch, so resolving its own directory here
# is safe and gives run_conversion.sh's real location for free.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_NAME="$(basename "$OUTPUT_DIR")"

job_ids=()
chunk_start=$FIRST
while [ "$chunk_start" -le "$LAST" ]; do

    chunk_end=$(( chunk_start + CHUNK_SIZE - 1 ))
    if [ "$chunk_end" -gt "$LAST" ]; then
        chunk_end=$LAST
    fi

    output_file="$OUTPUT_DIR/pressure_frames_$(printf '%04d' "$chunk_start")_$(printf '%04d' "$chunk_end").h5"
    job_name="pchunk_${CASE_NAME}_${chunk_start}_${chunk_end}"

    submit_out=$(sbatch --job-name="$job_name" --time="$TIME_PER_CHUNK" \
        --output="$OUTPUT_DIR/${job_name}_%j_out.txt" --error="$OUTPUT_DIR/${job_name}_%j_err.txt" \
        "$SCRIPT_DIR/run_conversion.sh" pressure "$SNC_PATH" "$output_file" \
        --first "$chunk_start" --last "$chunk_end" "${EXTRA_ARGS[@]}")

    new_jobid=$(echo "$submit_out" | grep -oE '[0-9]+$')
    echo "chunk [$chunk_start,$chunk_end] -> $output_file : job $new_jobid"
    job_ids+=("$new_jobid")

    chunk_start=$(( chunk_end + 1 ))
done

echo
echo "Submitted ${#job_ids[@]} chunk jobs: ${job_ids[*]}"
echo "Track progress with: squeue -u \$USER"
