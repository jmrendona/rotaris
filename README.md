# rotaris — `.snc` extraction pipeline

Two data sources, each used only for what's been validated:

## 1. Forces, Skin Friction, Normals, Geometry → `SNCReader` (raw `.snc`)

`converters/snc_reader.py` reads a PowerFLOW surface measurement `.snc`
file directly (it's NetCDF) and converts it to a compact HDF5.

- `Surface X/Y/Z-Force` and `Skin Friction` are converted to physical units
  (Pa) automatically in `to_h5()`, using the file's own lattice scale
  factors. Validated against the file's own stored `Skin Friction`:
  correlation ~0.999.
- Surfel normals, centroids (positions), areas, and rotor axis/RPM
  metadata come along with it.
- `Static Pressure` is deliberately **not** converted here (see below) -
  it's written raw, flagged `physical_units=False`.

No cluster-side extraction step is needed for this - the `.snc` file
itself is the input:

```python
from converters.snc_reader import SNCReader

reader = SNCReader("forces_rotor.snc")
reader.to_h5("forces_rotor.h5")
```

## 2. Static Pressure → `pf2ens` (do not derive it from raw `.snc`)

PowerFLOW translates static pressure between lattice and real units through
an internal Cp-based mechanism that could not be reverse-engineered from
the file's own stored scale factors - two attempts both failed (wrong
absolute level, and separately wrong spatial/chordwise shape). `pf2ens`
does this conversion correctly and is the only source trusted for pressure.

### Running this on the cluster

**You do not run `pf2ens` yourself.** `convert_snc_to_h5()` (in
`converters/ensight_to_h5.py`) calls it for you internally, once per
frame, and deletes each frame's intermediate EnSight export (~800 MB)
before moving to the next one, so nothing piles up on disk. All you run
is this one Python function (or the equivalent CLI command).

**Prerequisites**: `pf2ens` must be on `$PATH` - it already is inside
the PowerFLOW module environment you'd normally load to run/post-process
a case on the cluster. The Python environment needs `h5py`, `numpy`,
`scipy`, and `pyvista` installed.

Python:

```python
from converters.ensight_to_h5 import convert_snc_to_h5

convert_snc_to_h5(
    snc_path='pressure_rotor.snc',    # input: raw PowerFLOW measurement file
    output_path='pressure_rotor.h5',  # output: one combined HDF5 file, all frames
    first_frame=0,                    # first frame index to convert (inclusive)
    last_frame=199,                   # last frame index to convert (inclusive)
    surface_split=True,               # optional: also split into Upper/Lower groups
)
```

Or the same thing from the command line, no Python script needed:

```bash
python converters/ensight_to_h5.py pressure_rotor.snc pressure_rotor.h5 \
    --first 0 --last 199 --surface-split
```

What each piece means:
- `snc_path` / `output_path`: input `.snc` and output `.h5` paths.
- `--first` / `--last` (or `first_frame`/`last_frame`): inclusive frame
  range to convert - has to match frames that actually exist in the
  file. Check with `SNCReader(snc_path).n_frames` if unsure.
- `--surface-split`: optional. Splits into `Upper`/`Lower` groups (see
  below) - opens `snc_path` a second time internally to borrow its
  validated classification; no extra input file needed from you.
- `--nc-stats`: optional. If you've separately saved
  `exaritool nc-stats.ri <file>.snc -detail > nc_stats.txt`, pass
  `--nc-stats nc_stats.txt` to also fill in real per-frame timing and
  `LRF_position(rad)` in the output's `Metadata` group. Without it,
  those fields are left blank.
- `--work-dir`: optional, where intermediate `pf2ens` exports are
  written per frame (default: an auto-cleaned temp directory).
- `--reference-frame`: optional, which frame's geometry gets stored
  (default: `--first`).

For inspecting a single already-extracted frame by hand (e.g. while
debugging), you can still call `pf2ens` directly and read the result:

```bash
pf2ens -f <frame> -b frame_<frame> <pressure_measurement>.snc
```

```python
from converters.ensight_to_h5 import EnsightFrame

frame = EnsightFrame("frame_0.case")
pressure_pa = frame.variable('Static Pressure')
```

## Using both together (see manager.py)

```python
from converters.snc_reader import SNCReader
from converters.ensight_to_h5 import EnsightFrame

forces_reader = SNCReader('forces_rotor.snc')
forces_reader.to_h5('forces_rotor.h5')

pressure_frame = EnsightFrame('frame_0.case')
pressure_pa = pressure_frame.variable('Static Pressure')
```

**Important**: positions from the two sources are *not* index-matched -
`pf2ens` re-triangulates some surfels for EnSight compatibility (different
point/cell counts). Use each source's own geometry for its own variables;
don't mix per-point data across the two.

## Running on the cluster: `convert.py` / `run_conversion.sh`

`convert.py` is a CLI wrapping both branches above, one subcommand each:

```bash
python convert.py forces   <snc_path> <output.h5> [--face-name NAME] [--surface-split]
python convert.py pressure <snc_path> <output.h5> --first N --last M [--surface-split] [--nc-stats FILE] [--reference-frame N] [--work-dir DIR]
```

(`manager.py` is separate - ad-hoc, per-case scripting for whatever
conversion/post-processing is being worked on at the time, not meant to be
run as-is. `convert.py` is the stable, general entry point.)

`run_conversion.sh` submits either subcommand as a single-node SLURM batch
job (nothing here is parallelized, so it only ever requests one node/one
task - see the comment at its top before changing that):

```bash
sbatch run_conversion.sh forces   /path/case.snc /path/out.h5 --surface-split
sbatch run_conversion.sh pressure /path/case.snc /path/out.h5 --first 0 --last 199 --surface-split
```

Everything after `run_conversion.sh` on the command line is forwarded
straight to `convert.py`.

**One-time setup, before the first `sbatch` submission**: Alliance compute
nodes have no internet access, so the Python venv `run_conversion.sh`
expects (`~/rotaris-venv`) has to be built from a **login node** first:

```bash
module load StdEnv/2023 python/3.11 scipy-stack/2023b vtk
virtualenv --no-download --system-site-packages ~/rotaris-venv
source ~/rotaris-venv/bin/activate
pip install --no-index --upgrade pip
pip install pyvista
```

Use `virtualenv --no-download`, not `python -m venv` - Alliance's `python`
module ships without `ensurepip` bundled, so `python -m venv` fails trying
to fetch pip from the internet during creation itself
(`Error: Command '[...ensurepip...]' returned non-zero exit status 1`).
`virtualenv` knows to skip that and pull pip from Alliance's own local wheel
mirror instead (`pip install --no-index --upgrade pip`). `pyvista` turned
out to already be in that same local mirror (cvmfs wheelhouse), so this
whole block actually runs fine without real internet either way - it's
still meant to run once from a login node, though, since compute nodes
aren't guaranteed the same wheelhouse access. `run_conversion.sh` will
refuse to run (with a clear message) if `~/rotaris-venv` doesn't exist yet,
rather than trying to build it inside the batch job.

After that, `run_conversion.sh` just reuses the venv.

**Where things live**: `run_conversion.sh` locates `convert.py` via its own
path, not your current directory - so the `rotaris/` folder (code) can sit
anywhere, e.g. `$HOME`, and you can `sbatch ~/rotaris/run_conversion.sh ...`
from wherever you want the job's data/logs to land (typically `$SCRATCH` or
your `/project` allocation), no need to `cd` into the code folder first.
SLURM's `--output`/`--error` logs land in whatever directory you ran
`sbatch` from, and `<snc_path>`/`<output>` are resolved relative to that
same directory (or use absolute paths to be unambiguous).

`--time`/`--mem`/`--cpus-per-task` at the top of `run_conversion.sh` are
placeholder guesses, not measured against a real case yet - adjust once
you've seen actual usage.

## Splitting into upper/lower surface

Needed for anything that requires knowing which side of the blade a point
is on (e.g. skin-friction lines). Position-based splitting (e.g. `Y > 0`)
is unreliable near the leading/trailing edges, where thickness goes to
zero - validated: two independent methods agree only ~55% of the time in
that zone (a coin flip).

- **`SNCReader.surface_split()`**: sign of the surfel normal's component
  along the rotation axis (`lrf_axis_direction`). Validated: its own
  ambiguous zone is ~45x narrower than position-based. This is the
  trusted classification - use it directly for raw-`.snc`-derived data
  (`to_h5(..., surface_split=True)` writes separate `Geometry/Upper`,
  `Geometry/Lower`, `Data/Upper`, `Data/Lower` groups).
- **`EnsightFrame.surface_split()`**: do **not** recompute this from
  pf2ens's own mesh - `pf2ens` splits complex surfels into quads/trias
  for EnSight compatibility, which breaks mesh connectivity enough that
  VTK's `compute_normals()` becomes locally inconsistent (validated:
  ~52% agreement with ground truth, i.e. useless, even away from the
  edges). Instead, classification is borrowed from the raw `.snc` file
  via nearest-neighbor position matching:

  ```python
  from converters.snc_reader import SNCReader
  from converters.ensight_to_h5 import raw_positions_to_ensight_frame

  reader = SNCReader('forces_rotor.snc')
  ref_positions = raw_positions_to_ensight_frame(
      reader.surfel_centroids() * reader.lattice_scales['LatticeLength']
  )
  ref_upper = reader.surface_split()
  ```

  `raw_positions_to_ensight_frame()` matters: pf2ens centers each axis on
  the mesh's own bounding-box midpoint, not on `lrf_axis_origin` - the two
  only coincide (by symmetry) for the axes perpendicular to the rotation
  axis. Skipping this re-centering silently breaks the nearest-neighbor
  match on the axis-aligned coordinate.

  `convert_snc_to_h5(..., surface_split=True)` (and `--surface-split` on
  the CLI) does all of this automatically.

## Output file structure

### `SNCReader.to_h5()` - all frames in the file, one `.h5` file

Geometry (positions/normals/areas) is written once - the raw `.snc` file
carries no frame axis for it, only for `measurements` - and every
variable is written as a 2D dataset covering every frame in the file:

```
Metadata/             lrf_axis_origin (meters), lrf_axis_direction, frame_index
                       (datasets, frame_index = arange(n_frames)); scale_*,
                       offset_*, lrf_angular_vel_lattice (attrs)
Geometry/X,Y,Z         surfel centroid positions, meters (1D, shape (n_points,))
Geometry/NX,Normal_Y,Normal_Z, Area   normals (unit vectors) and area, m^2
Data/<Variable_Name>   one dataset per .snc variable (spaces -> underscores),
                       shape (n_frames, n_points), each with attrs
                       lattice_unit_class, physical_units
```

(Positions/area/`lrf_axis_origin` are scaled by `LatticeLength` before being
written - fixed 2026-08-08, `to_h5()` used to write these in raw lattice
units while `Data/<Variable_Name>` was already physical, an inconsistency
nothing downstream had started depending on yet, found while building
`bladeprocessor/friction_lines.py`, the first real consumer of this file's
`Geometry` group.)

With `surface_split=True`, `Geometry` and `Data` are each replaced by
`Geometry/Upper`, `Geometry/Lower`, `Data/Upper`, `Data/Lower` (same
datasets underneath, just partitioned by surfel). `n_frames` is read
directly from the file (`reader.n_frames`, from `measurements`' first
axis) - no argument needed, `to_h5()` always writes every frame present.

Validated on a real 2-frame file: `Data/Upper/Skin_Friction` came out
shape `(2, 13319966)`, frame 0's mean matched the earlier single-frame
result exactly (9.6861 N/m²), frame 1 close by as expected for a
consecutive timestep.

### `EnsightSeriesWriter` / `convert_snc_to_h5()` - multi-frame

**One `.h5` file total, not one per frame and not one group per frame.**
Frames are rows in shared 2D datasets:

```
Geometry/X,Y,Z                 written ONCE, from the is_reference=True
                                frame (1D, shape (n_points,)) - no
                                normals (see below)
Data/<variable>                one 2D dataset per variable,
                                shape (n_frames, n_points) - each
                                add_frame() call appends a row
Metadata/frame_index, start_ts, end_ts, mid_ts, mid_s, lrf_position_rad
                                1D, shape (n_frames,) - row i describes
                                Data's row i
```

With `surface_split=True`: `Geometry/Upper`, `Geometry/Lower`,
`Data/Upper/<variable>`, `Data/Lower/<variable>` (same 2D-rows-per-frame
shape underneath).

**No normals are written here**, on purpose: this writer exists for
`pf2ens`-derived variables (chiefly `Static Pressure`), which don't need
normals, and `EnsightFrame.normals()` isn't reliable enough to propagate
downstream for anything else - see the section above and
`EnsightFrame.normals()`'s docstring. Upper/lower classification always
comes from `SNCReader.surface_split()` (raw `.snc` normals) instead.

Storing geometry once assumes the blade is rigid (only orientation
changes between frames, not shape) - noted as an assumption in
`EnsightSeriesWriter`'s docstring, with per-frame geometry storage
mentioned as the more foolproof (but more storage-hungry) alternative,
not implemented.

## Wall shear / friction lines: `bladeprocessor/FrictionLines` - Equations

`FrictionLines` consumes a `SNCReader.to_h5(..., surface_split=True)` file
(the forces branch above - never the `pf2ens`/pressure branch, whose
recomputed normals aren't trustworthy for this, see "Splitting into
upper/lower surface"). Every quantity below is per-surfel unless noted.

**Wall shear vector** (`wall_shear()`) - the surface force with its
normal (pressure) component removed, leaving only the tangential
(friction) part:

```
tau = F - (F . n) n
```

`F` is `Surface X/Y/Z-Force` (already Pa, from `SNCReader.to_h5()`), `n`
is this surfel's own unit normal (`Geometry/NX,Normal_Y,Normal_Z`, from
the raw `.snc` - not `pf2ens`'s recomputed one).

**Skin friction coefficient** (`cf()`) - normalized by a LOCAL dynamic
pressure, using each surfel's own radius, not one fixed velocity for the
whole blade:

```
Cf = tau / q_ref
q_ref = 0.5 * rho_ref * (omega * r)^2
omega = rpm * 2*pi / 60
```

`r` is this surfel's physical radius from the rotor's rotation axis
(`lrf_axis_origin`/`lrf_axis_direction`, the one place `FrictionLines`
still uses the rotation axis rather than raw Cartesian position - see
`_radius()`). This matches `BladePostProcessor.compute_cf()` elsewhere in
this project (`U_ref(r) = omega * r`), rather than
non-dimensionalizing by one global freestream/tip velocity. **Caveat**:
`q_ref -> 0` as `r -> 0`, so `Cf` blows up near the rotation axis - some
faces (e.g. `Rotor::Default-Segment`) include a sliver of hub/bore
geometry right at `r~0`; exclude it (`span_min`/`span_max` in
`friction_lines()`, or just don't query `cf_at_radii()` near `r=0`)
before trusting an unrestricted `cf()` call.

`component` picks which part of `tau` to report:

```
Cf (magnitude)  = |tau| / q_ref                    >= 0 always
Cf (chordwise)  = tau[chord_axis] / q_ref           signed
Cf (spanwise)   = tau[span_axis]  / q_ref           signed
```

Magnitude can never show a sign reversal (separation/reattachment) -
it just dips toward zero. The signed components can, which is the
point of having them (see `friction_lines_test_cf_chordwise_*.png`
crossing zero mid-chord at the outer radii).

**Local chordwise position** (`cf_at_radii()`) - for a thin band of
surfels around a target radius (`|r - r_target| < tol`), the raw
Cartesian chord coordinate (`chord_axis`, centered - see `_span_chord()`)
is rescaled to `[0, 1]` using THAT BAND's own min/max:

```
x/c = (chord - chord_min) / (chord_max - chord_min)
```

This is a per-band, per-case rescaling, not a case-independent x/c (see
the class docstring's caveat and README's "What's still open" below).
The resulting `(x/c, Cf)` pairs are then averaged into `n_chord_bins`
equal-width x/c bins (mean Cf per bin) - without this, a single radius
band on a real `.snc` surface contains hundreds of thousands of raw,
noisy points, which reads as a dense cloud rather than a curve once
plotted; the point of `n_chord_bins` is turning that cloud into the
single readable curve per radius that `plot_cf_radii()` draws.

## What's still open

- Iso-radius / (r/R, x/c) resampling directly from raw `.snc` surfel
  clouds (needed for skin-friction-line or Cp-vs-chord plots) isn't solved
  yet - see `bladeprocessor/` for the grid-based tools that work on
  already-resampled `SurfaceField`-style HDF5 files instead.
- The additive constant for converting raw lattice `Static Pressure` to Pa
  remains unresolved and is not assumed to be case-independent - hence the
  hard requirement to use `pf2ens` for pressure rather than the raw file.
- `FrictionLines`' span/chord axes are raw Cartesian columns
  (`span_axis`/`chord_axis`), not derived from the rotor's rotation axis -
  a rotation-axis-based derivation was tried and abandoned (see the class
  docstring) because it assumes the chord line lies in the rotor disk
  plane, which breaks on any blade with real geometric pitch/twist. The
  Cartesian fallback works for every case in this project so far but
  isn't automatically correct for a differently-oriented mesh, and
  doesn't by itself handle a twisted blade's true local chord direction
  either - a proper fix (e.g. per-station PCA of the point cloud) is not
  implemented.
