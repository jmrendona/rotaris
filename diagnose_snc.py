"""
Diagnostic for the open "Cf grows ~25x with frame index" investigation -
see HANDOFF.md's "OPEN INVESTIGATION" section for full context.

Checks two things against a real, multi-frame .snc file:
  1. Whether this file even needs the patched large-record NetCDF reader
     (_LargeRecordNetcdfFile) at all - if plain, unmodified scipy can
     open it too, that gives a direct, no-ambiguity A/B comparison.
  2. Whether the frame-to-frame growth already shows up in the RAWEST
     possible read (straight out of SNCReader.variable(), no HDF5
     round-trip, no FrictionLines math) - and whether our patched reader
     and plain scipy agree on it.

If section 2 and section 3 (plain scipy) match: our reader is exonerated
- the growth is really in the stored data, and this becomes a
simulation-convergence question, not a code bug.
If they DON'T match: _LargeRecordNetcdfFile has a real bug on this
file's specific layout that none of the earlier synthetic tests (single
record var, two record vars, byte-padding case - see snc_reader.py's
class docstring) happened to exercise - needs a fix, not a workaround.

Usage: python diagnose_snc.py /path/to/case.snc
"""
import sys
import numpy as np
import scipy.io as sio
sys.path.insert(0, '.')  # run from the rotaris repo root
from converters.snc_reader import SNCReader, _LargeRecordNetcdfFile  # noqa: F401

path = sys.argv[1]
frames_to_check = [0, 1, 10, 40, 80, 150]

print("=== 1. Does plain, unmodified scipy open this file at all? ===")
try:
    f_plain = sio.netcdf_file(path, mmap=False)
    print("plain scipy: OK - this file does NOT need the large-record fix.")
    plain_ok = True
except Exception as e:
    print(f"plain scipy FAILED ({type(e).__name__}: {e}) - this file needs the patched reader, "
          "can't A/B against plain scipy directly.")
    plain_ok = False

print("\n=== 2. Raw per-frame magnitude via SNCReader (our patched reader) ===")
reader = SNCReader(path)
print("n_frames:", reader.n_frames)
print("variable_names:", reader.variable_names)

# Adjust this name if 'Skin Friction' isn't in variable_names - the print() above shows the exact list.
name = 'Skin Friction' if 'Skin Friction' in reader.variable_names else reader.variable_names[0]
print(f"using variable: '{name}'")

for frame in frames_to_check:
    if frame >= reader.n_frames:
        continue
    v = reader.variable(name, frame=frame)
    mag = np.linalg.norm(v, axis=-1) if v.ndim > 1 else np.abs(v)
    flat = np.asarray(v).reshape(v.shape[0], -1) if v.ndim > 1 else np.asarray(v).reshape(-1, 1)
    print(f"  frame {frame:4d}: mean|.|={mag.mean():.6g}  max|.|={mag.max():.6g}  "
          f"first 3 raw values={flat[:3].ravel()[:3]}")

if plain_ok:
    print("\n=== 3. Same thing via PLAIN scipy directly (bypassing our fix entirely) ===")
    idx = reader.variable_index[name]
    for frame in frames_to_check:
        if frame >= reader.n_frames:
            continue
        v = f_plain.variables['measurements'][frame, idx, :]
        mag = np.abs(v)
        print(f"  frame {frame:4d}: mean|.|={mag.mean():.6g}  max|.|={mag.max():.6g}  "
              f"first 3 raw values={np.asarray(v)[:3]}")
    print("\nIf section 2 and section 3 match: our reader is exonerated, this is a real "
          "simulation-convergence issue, not a bug in the code.")
    print("If they DON'T match: our patched reader has a real bug on this file - report back "
          "with both outputs.")
else:
    print("\nPlain scipy can't open this file, so we can't A/B directly - but the per-frame "
          "trend in section 2 (does it grow the same way as the Cf plots?) still tells us "
          "whether the growth is present at the rawest possible read.")
