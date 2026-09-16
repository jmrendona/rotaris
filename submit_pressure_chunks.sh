#!/bin/bash

# Submit ONE SLURM job array covering a whole `pressure` frame range,
# split into fixed-size chunks (one array TASK per chunk - see
# run_pressure_chunk_array.sh, which each task actually runs), throttled
# to at most --max-concurrent chunks running AT THE SAME TIME
# (`--array=0-N%<max_concurrent>` - SLURM's own built-in concurrency
# limit: it queues the rest and starts the next one the instant a
# running slot frees up, whether that task finished or died, no
# resubmitting or manual batch-tracking needed).
#
# WHY THE THROTTLE, NOT JUST "SUBMIT EVERYTHING": convert_snc_to_h5()
# calls pf2ens once per frame, and each call needs one seat from this
# account's shared, LIMITED PowerFLOW license pool
# (exasignalprocessingjob - check current usage with whatever command
# your PowerFLOW env provides for license status). An earlier version of
# this pipeline submitted every chunk as an independent job with no
# throttle at all - confirmed on a real 26-chunk run that several chunks
# died individually ~2 minutes after their own start, each killed by an
# automated system process (UID 0 in `sacct`), consistent with too many
# concurrent pf2ens processes exceeding available license seats at that
# instant and getting reaped after a grace period. Throttling concurrency
# keeps peak license demand bounded and leaves headroom for other users/
# jobs on the same shared license pool - --max-concurrent is a real
# tuning knob, not a formality; start conservative (e.g. 5) and raise it
# only after confirming no chunks are dying this way at that level.
#
# Writes ONE output .h5 PER CHUNK
# (<output_dir>/pressure_frames_<first>_<last>.h5) - convert_snc_to_h5()
# always creates a fresh file (h5py.File(path, 'w')), it can't append to
# an existing one, so a single shared output file across chunks was
# never an option. Merging the chunk files into one afterward is a
# SEPARATE step, not done here.
#
# Usage:
#   ./submit_pressure_chunks.sh <snc_path> <output_dir> \
#       --first N --last M --chunk-size S --time-per-chunk HH:MM:SS --max-concurrent K \
#       [extra convert.py args...]
#
# <snc_path>/<output_dir> are positional (always required, same as
# convert.py's own snc_path/output); --first/--last/--chunk-size/
# --time-per-chunk/--max-concurrent are named flags, matching this
# project's own convert.py convention (--first N --last M, etc.) rather
# than yet another fixed positional order to memorize - all five are
# REQUIRED, no defaults: the right chunk size/time budget/concurrency is
# genuinely case-specific (per-frame cost varies a lot file to file), so
# a silent default risks quietly reproducing the exact time-limit or
# license-contention problems this script exists to avoid. Anything else
# not recognized above (e.g. --surface-split, --face-names ...) is
# forwarded as-is to each chunk's own `run_conversion.sh pressure` call.
#
# Example (this session's actual case - 773 frames, 30/chunk -> 26
# chunks, 12h/chunk budget against an expected ~8.1h/chunk at this
# file's own measured rate, throttled to 5 running at once):
#   ./submit_pressure_chunks.sh \
#       /scratch/jmrendon/Rotor-alone/6e-5_6000rpm_HF/SMF_fwh_rotor.snc \
#       /scratch/jmrendon/Rotor-alone/6e-5_6000rpm_HF/pressure_chunks \
#       --first 0 --last 772 --chunk-size 30 --time-per-chunk 12:00:00 --max-concurrent 5 \
#       --surface-split

set -euo pipefail

usage() {
    echo "Usage: $0 <snc_path> <output_dir> --first N --last M --chunk-size S " \
         "--time-per-chunk HH:MM:SS --max-concurrent K [extra convert.py args...]" >&2
    exit 1
}

if [ "$#" -lt 2 ]; then
    usage
fi

SNC_PATH="$1"; OUTPUT_DIR="$2"
shift 2

FIRST=""; LAST=""; CHUNK_SIZE=""; TIME_PER_CHUNK=""; MAX_CONCURRENT=""
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --first) FIRST="$2"; shift 2 ;;
        --last) LAST="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --time-per-chunk) TIME_PER_CHUNK="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$FIRST" ] || [ -z "$LAST" ] || [ -z "$CHUNK_SIZE" ] || [ -z "$TIME_PER_CHUNK" ] || [ -z "$MAX_CONCURRENT" ]; then
    echo "Missing one or more required flags (--first/--last/--chunk-size/--time-per-chunk/--max-concurrent)." >&2
    usage
fi

mkdir -p "$OUTPUT_DIR"

# NOT ${BASH_SOURCE[0]}'s directory alone - this script (unlike
# run_conversion.sh / run_pressure_chunk_array.sh) is run directly from
# a login node, not copied to a compute-node spool dir by sbatch, so
# resolving its own directory here is safe and gives the array script's
# real location for free.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

n_chunks=$(( (LAST - FIRST) / CHUNK_SIZE + 1 ))
max_index=$(( n_chunks - 1 ))

echo "Submitting $n_chunks chunk(s) covering frames [$FIRST,$LAST] (size $CHUNK_SIZE), " \
     "at most $MAX_CONCURRENT running at once."

submit_out=$(sbatch --array="0-${max_index}%${MAX_CONCURRENT}" --time="$TIME_PER_CHUNK" \
    --output="$OUTPUT_DIR/pchunk_%A_%a_out.txt" --error="$OUTPUT_DIR/pchunk_%A_%a_err.txt" \
    "$SCRIPT_DIR/run_pressure_chunk_array.sh" "$SNC_PATH" "$OUTPUT_DIR" "$FIRST" "$CHUNK_SIZE" "$LAST" \
    "${EXTRA_ARGS[@]}")

echo "$submit_out"
array_jobid=$(echo "$submit_out" | grep -oE '[0-9]+$')

echo
echo "Array job $array_jobid submitted - $n_chunks tasks (indices 0-$max_index), throttled to $MAX_CONCURRENT concurrent."
echo "Track progress with: squeue -u \$USER"
echo "Per-task logs: $OUTPUT_DIR/pchunk_${array_jobid}_<task_index>_out.txt (and _err.txt)"
