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

## 3. Volumetric planes/cuts → `.fnc` (pf2ens + pyvista, see `converters/fnc_plane.py`)

`.fnc` files are PowerFLOW's *volumetric* fluid measurement (as opposed to
`.snc`'s surface measurement) - much larger, multi-resolution octree data.
Same rule as pressure above: this goes through `pf2ens` (per frame) +
`pyvista` sampling, never PowerVIZ, never a direct `.fnc` read.

Two extraction "shapes" are what you'll normally want:

- **`fnc-meridional`**: a single rotated hub-to-tip, inlet-to-outlet plane
  at one fixed azimuth angle (e.g. "the plane through 0°").
- **`fnc-iso-radius`**: a fixed-radius cylindrical cut, unrolled into an
  azimuth-vs-axial-position 2D grid (e.g. "the cut at 85% span").

Both take a `--variables` list of `pf2ens` short codes (run
`pf2ens -d <fnc_path>` against your file to see what's available - e.g.
`vmag,p,vx,vy,vz`) and a `--first`/`--last` frame range.

### Masking - what actually works, and what does not

**`--freeze-mask-variable vmag`** is the one masking option that is
validated and safe to use. PowerFLOW never updates lattice cells inside
solid, body-fixed geometry (e.g. the stator hub) - across frames their
value barely changes, unlike real flow. This flag computes `Data/frozen`
from that near-zero-variance signature and needs `--first`/`--last` to
span at least 2 (ideally several, spread out) frames to be reliable.
`fnc_plane.plot_frame()` (and `fnc-plot` below) automatically combines
`Data/frozen` into the valid-point mask if it's present - you don't have
to do anything extra to use it once it's in the file.

**There is no working mask for the rotor blades.** `vtkValidPointMask`
(pf2ens/pyvista's own solid-geometry flag) does not track the blades at
all near mid-to-tip span - confirmed empirically, not assumed. A real,
axially-aware blade mask was attempted (see `RotorBladePosition` in
`fnc_plane.py`) and reverted after several rounds because it never held
up as a trustworthy, full-plot-scale result. What *is* validated and
available is **identification only**: `RotorBladePosition.blade_azimuths_deg()`
draws a line at each blade's true instantaneous azimuth (matched against
the real wake hot-spot across ~190° of rotation) - useful as a visual
overlay so you know where a blade is in a plot, but it does not remove or
flag any data. This is Python-API only, not wired into the CLI:

```python
from converters import fnc_plane

blades = fnc_plane.RotorBladePosition('SMR-VR8.snc')  # rotor .snc, once
# lrf_position_rad for a given frame comes from your extracted .h5's
# Metadata/lrf_position_rad[frame_index]
azimuths = blades.blade_azimuths_deg(lrf_position_rad=1.63)

fnc_plane.plot_frame(
    'iso_r85pct.h5', 'vmag', frame_index=0,
    savepath='iso_r85pct_frame0_bladeoverlay.png',
    blade_azimuths_deg=azimuths,   # only valid for kind='iso_radius' files
)
```

### Commands

```bash
# one meridional plane at 0 degrees azimuth, frames 0-99, with stator-hub masking
python convert.py fnc-meridional SMR-VR8.fnc plane_0deg.h5 \
    --angle 0 --variables vmag,p --first 0 --last 99 \
    --freeze-mask-variable vmag \
    --plot plane_0deg_frame0.png --plot-frame 0

# one iso-radius cut at r=0.274m (85% span for this case), frames 0-99, with masking
python convert.py fnc-iso-radius SMR-VR8.fnc iso_r85pct.h5 \
    --radius 0.274 --variables vmag,p --first 0 --last 99 \
    --freeze-mask-variable vmag

# several meridional planes in one command (shares one nc-stats.ri call)
python convert.py fnc-meridional-sweep SMR-VR8.fnc planes_out/ \
    --angle-start 0 --angle-end 350 --angle-step 10 \
    --variables vmag,p --first 0 --last 99 --freeze-mask-variable vmag

# quick-look plot from an already-extracted file (no re-extraction)
python convert.py fnc-plot iso_r85pct.h5 vmag iso_r85pct_frame0.png --frame 0
```

`--first`/`--last` must span >= 2 frames (spread out, not adjacent) for
`--freeze-mask-variable` to calibrate correctly - see `fnc_plane.py`'s
notes on threshold calibration if the default `--freeze-rel-threshold
0.01` over- or under-masks for your case. `fnc-freeze-mask` recomputes
`Data/frozen` on an existing file post-hoc if you need to retune this
without re-running the (expensive) `pf2ens` extraction.

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
python convert.py fnc-meridional <fnc_path> <output.h5> --angle DEG --variables v1,v2 --first N --last M [--freeze-mask-variable vmag]
python convert.py fnc-iso-radius <fnc_path> <output.h5> --radius M --variables v1,v2 --first N --last M [--freeze-mask-variable vmag]
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
sbatch run_conversion.sh fnc-meridional /path/case.fnc /path/plane_0deg.h5 \
    --angle 0 --variables vmag,p --first 0 --last 99 --freeze-mask-variable vmag
sbatch run_conversion.sh fnc-iso-radius /path/case.fnc /path/iso_r85pct.h5 \
    --radius 0.274 --variables vmag,p --first 0 --last 99 --freeze-mask-variable vmag
```

**Use `sbatch`, not a direct shell command, for any `fnc-*` extraction** -
`.fnc` files are far larger than `.snc` (100s of GB to multi-TB) and each
frame's `pf2ens` export/sample is real, sustained CPU+I/O work, not
something to run on a login node. `fnc-plot` and `fnc-freeze-mask` are the
only exceptions worth running directly in a shell (after loading the venv
per below) - they only touch an already-extracted, comparatively small
`.h5` file, no `pf2ens`/`.fnc` access at all.

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
Metadata/lrf_axis_origin, lrf_axis_direction
                                frame-independent (rotation axis doesn't
                                change), meters - matches SNCReader.to_h5()'s
                                schema, always populated by
                                convert_snc_to_h5(). lrf_axis_origin is
                                re-centered onto pf2ens's own bounding-box
                                convention (same shift Geometry/X,Y,Z
                                already gets - lrf_axis_origin is NOT
                                pf2ens's coordinate origin, see
                                raw_positions_to_ensight_frame()), which
                                needs reading the full raw surfel cloud
                                once, even when surface_split=False
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

**Two bugs found and fixed in `cf_at_radii()`/`plot_cf_radii()`** (this
project's own `2f_SMF_forces_rotor.snc` case, a 2-bladed rotor with both
blades lumped into a single `/Rotor::Default-Segment` face):

1. **Two-blade chord mixing.** `cf_at_radii()` originally had no
   `span_min`/`span_max` (unlike `friction_lines()`), so a radius band
   picked up both blades' surfels at once (span runs symmetrically from
   -0.125 to +0.125 m here, radius ~= `|span|`, so `r_target` matches
   both `span=+r_target` and `span=-r_target`) - normalizing x/c against
   their COMBINED chord range produced a spurious extra peak at both
   x/c=0 AND x/c=1 instead of once, near the real leading edge (see
   `friction_lines_test_cf_mag_avg.png`'s before/after - confirmed
   against `images/cf/skin-friction_radii_plot.png`, a trusted reference
   with a single clean peak). Fixed by adding the same `span_min`/
   `span_max` cropping `friction_lines()` already had; this case needed
   `span_min=0.02` to clear its hub-region point cluster (found via a
   histogram of `_span_chord()`'s span - no reliable automatic value,
   check per case).
2. **Flipped LE/TE orientation.** Even after fixing (1), the peak showed
   up at x/c=1 instead of x/c=0 - the same orientation ambiguity already
   documented for Cp (`SurfaceVariable.at_radii()`'s `reverse_chord`):
   x/c=0 is arbitrarily assigned to whichever raw chord extreme happens
   to be the minimum value, with no inherent LE/TE meaning. Added the
   same `reverse_chord` parameter here; this case needed
   `reverse_chord=True` to match the trusted reference (Cf peaking
   sharply near x/c=0, decaying toward x/c=1).

`plot_cf_radii()` also now defaults to `cividis` (not `viridis`) and
draws a grid, matching `SurfaceVariable`'s radius-colored plots.

## Any surface variable at radii: `bladeprocessor/SurfaceVariable`

Generalizes `FrictionLines.cf_at_radii()`/`plot_cf_radii()` beyond wall
shear/Cf to **any** named variable stored under `Data/<surface>/<name>` -
Cp, `y+`, RMS statistics of anything, or a raw variable as-is. Works
against either data branch's HDF5 output, since both share the same
schema (`Geometry/Upper,Lower` positions, `Data/Upper,Lower/<var>` with a
frame axis, `Metadata/lrf_axis_origin,lrf_axis_direction`):

- `SNCReader.to_h5(..., surface_split=True)` (forces branch)
- `convert_snc_to_h5(..., surface_split=True)` (pressure branch)

**Not** for wall shear/Cf itself - that needs normals and the
`tau = F - (F.n)n` derivation, still `FrictionLines`' job. Span/chord/
radius conventions (raw Cartesian `span_axis`/`chord_axis`, rotation-
axis-based radius) are identical to `FrictionLines` - same caveats apply
(assumes little blade twist).

```python
from bladeprocessor.surface_variable import SurfaceVariable

sv = SurfaceVariable('pressure_case.h5', r_tip=0.125, rho_ref=1.22523, rpm=6000, pref=101325)

# raw access to any stored variable - instantaneous, mean, or rms/raw_rms:
yplus_mean = sv.variable('y+', surface='Upper', frame=None, stat='mean')
yplus_frame0 = sv.variable('y+', surface='Upper', frame=0)  # stat ignored when frame is set

# Cp, normalized the same way FrictionLines.cf() normalizes Cf (see below):
cp_mean = sv.cp(surface='Upper', stat='mean')
cp_rms = sv.cp(surface='Upper', stat='rms')

# both surfaces at several radii, one plot:
sv.plot_cp_radii([0.045, 0.072, 0.100, 0.117, 0.122], stat='mean',
                  span_min=0.03, reverse_chord=True, savepath='cp_radii.png')
```

**Cp normalization** - identical convention to `FrictionLines.cf()`
(local `q_ref`, not one fixed velocity for the whole blade):

```
Cp = (p - pref) / q_ref
q_ref = 0.5 * rho_ref * (omega * r)^2
```

`pref` is only needed for `stat='mean'`/instantaneous - a constant offset
doesn't change a fluctuation's `rms`/`raw_rms`, so those work without it.

**Instantaneous vs. average vs. RMS** - every method takes the same
`frame`/`stat` pair `FrictionLines` uses: `frame=<int>` for one frame
directly (`stat` ignored - a statistic across frames isn't meaningful for
a single one), `frame=None` (default) reduces across every frame via
`stat` (`'mean'`, `'rms'` = fluctuation std about the mean, or
`'raw_rms'` = `sqrt(mean(x^2))`, includes any nonzero mean - see
`variable()`'s docstring for when you'd want which). Validated: the
`rms`/`raw_rms` reduction formulas match `np.std`/`sqrt(mean(x^2))`
exactly (synthetic check); only one real frame was available locally to
validate `stat='mean'`/instantaneous against real data, so `rms` itself
hasn't been checked against a real multi-frame case yet.

**Real bugs found and fixed while validating this against real Cp data**
(not data issues - all in the code):

- **Two-blade mixing**: a radius band on a file covering a whole
  multi-bladed rotor picks up every blade at that radius at once: `x/c`
  then gets normalized against the combined chord range of all of them,
  producing a garbled curve. Same issue `FrictionLines.friction_lines()`
  already had `span_min`/`span_max` for - `at_radii()`/`plot_at_radii()`/
  `plot_cp_radii()` now have the same parameters, with the same caveat:
  no reliable automatic value, pass what's right for your case's mesh
  (e.g. `span_min=0.03` for the case this was validated against).
- **x/c orientation**: the raw chord axis has no inherent leading/
  trailing-edge meaning - `x/c=0` was arbitrarily assigned to the minimum
  chord value, which came out backwards for the validated case (Kutta-
  condition-near-zero end at `x/c=0`, sharp LE stagnation+suction-peak
  signature at `x/c=1`). `reverse_chord=True` flips it - check per case
  (Cp should return to ~0 approaching the trailing edge and show the
  sharp stagnation/suction-peak feature at the leading edge; if that's at
  `x/c=1` instead, flip it).
- **Sign**: `plot_cp_radii()` labeled its y-axis `$-C_p$` but was
  plotting raw `+Cp`, unnegated - a real sign bug, not a display choice
  (fixed: `stat='mean'`/instantaneous now correctly negates; `rms`/
  `raw_rms` don't, since those are already non-negative and negating
  them would be wrong). **`BladePostProcessor.plot_radii()` elsewhere in
  this project has the identical bug** - labels `$-C_p$` but never
  negates - confirmed by reading that method directly, not fixed there
  (different class, out of scope here) - worth knowing if comparing
  against or reusing that tool's past output/figures.

**Plot style**: points (scatter), never a connected line, even when
`n_chord_bins` bin-averages the data (already smooth) - a line whose end
sits wherever a crop/percentile cut it off visually reads as "the curve
is missing" past that point; discrete points don't imply continuation.
Default colormap `cividis` (not `viridis`), one legend entry per radius
covering both surfaces (same color, two branches - matches this
project's established `-Cp`-radii plot style).

### Whole-blade surface plot: `plot_variable_surface()`

Generalizes `FrictionLines.friction_lines()` beyond Cf to any scalar
field (static pressure, Cp, `y+`, ...) - same style (dense scatter, not
interpolated onto a grid - see that method's docstring for why), one row
per surface, colors clipped to a percentile range (two-sided here, since
a general field like Cp can be negative, unlike Cf magnitude):

```python
sv.plot_variable_surface(
    lambda s: -sv.cp(surface=s, stat='mean'),  # note the negation - see "Sign" above
    cbar_label='-Cp', span_min=0.03,
    savepath='cp_surface.png',
)
```

An optional `get_vector` callable overlays a direction quiver (e.g. for a
surface velocity field), same as `friction_lines()`'s wall-shear quiver.

### Cross-case comparison: `to_common_grid()` / `field()` / `SurfaceVariableField`

Resamples any per-surfel field onto a shared `(r/R, x/c)` grid - same
purpose, convention, and two-stage algorithm (per-radius-band onto x/c,
then across bands onto r/R) as `SurfaceField.to_common_grid()`
(`bladeprocessor/surface_field.py`), built directly from a raw surfel
cloud instead of a pre-resampled `(Radius, Chord)` file. Only meaningful
for cases known to share the same geometry, or ones that are properly
scalable in both span and chord (same caveat `SurfaceField` already
carries).

Rather than reimplement comparison/delta plotting, `SurfaceVariable.field()`
wraps a `(SurfaceVariable, get_values)` pair as a `SurfaceVariableField` -
an object that duck-types `SurfaceField`'s interface (`to_common_grid()`,
`physical_aspect()`, `var_name`) closely enough to drop straight into the
existing `SurfaceFieldComparator`, **unmodified** - works interchangeably
against another `SurfaceVariableField` or an actual `SurfaceField`:

```python
from bladeprocessor.surface_field import SurfaceFieldComparator

field_2025 = sv_2025.field(lambda s: sv_2025.cp(surface=s, stat='mean'),
                            var_name='Cp 2025', c_ref=0.025, span_min=0.03)
field_2026 = sv_2026.field(lambda s: sv_2026.cp(surface=s, stat='mean'),
                            var_name='Cp 2026', c_ref=0.025, span_min=0.03)

comparator = SurfaceFieldComparator({'2025': field_2025, '2026': field_2026})
comparator.plot_cases(cbar_label='Cp', savepath='cp_comparison.png')
comparator.plot_delta('2025', '2026', cbar_label='Cp delta', savepath='cp_delta.png')
```

`c_ref` (unlike `SurfaceField`, which can fall back to `max(chord)` from
its own file) must be passed explicitly - raw surfel data has no native
chord axis to infer one from, and using the same physical `c_ref` across
every case being compared is what keeps `x/c` meaning the same thing in
each.

**Validated**: resampling a real Cp field and comparing it against
itself (same `SurfaceVariable`, same `get_values`) through the full
`SurfaceFieldComparator` pipeline gives exactly `0.0` delta wherever both
sides have data, and `plot_cases()` renders both (identical) panels
correctly - confirms the duck-typed integration works end-to-end.
`plot_delta()` itself hit a **pre-existing bug in `SurfaceFieldComparator`**
(`surface_field.py`, not part of this class): when a delta is exactly
zero everywhere (only possible in a degenerate self-comparison like this
test), `symmetric=True`'s `vmax = max(abs(delta)) = 0` makes
`contour_levels` a repeated `0`, and matplotlib rejects non-increasing
contour levels. Not hit by any real two-different-cases comparison, and
not fixed here (different class) - worth knowing if a genuinely-zero
delta ever comes up for real.

### Pressure fluctuation: `pressure_fluctuation()` / `plot_pressure_fluctuation()`

`p'(frame) = p(frame) - p_mean`, where `p_mean` is the mean over every
frame in the file - the same mean `variable(stat='rms')` uses internally
for `Prms = sqrt(mean(p'^2))`, so this fluctuation is consistent with
that statistic rather than some other baseline. Dimensional [Pa], not
normalized by `q_ref` (unlike `cp()`) - the fluctuation's own sign/shape
is the point here, not a cross-radius comparison.

```python
# one frame's fluctuation as a blade contour:
sv.plot_pressure_fluctuation(frame=0, span_min=0.03, savepath='p_fluct_frame0.png')

# one image per frame, e.g. for an animation:
for frame in range(sv.n_frames):
    sv.plot_pressure_fluctuation(frame, span_min=0.03, savepath=f'p_fluct_frame{frame:03d}.png')
```

`Prms` itself needs no new code - it's already `variable('static_pressure', stat='rms')`
(or `cp(stat='rms')` for the normalized version), usable directly with
`plot_variable_surface()`/`plot_at_radii()`.

**Validated**: on a real (single-frame) file, `pressure_fluctuation(0)`
returns exactly `0.0` everywhere, as expected (`p(frame) == p_mean` when
there's only one frame) - real signal will show up once compared against
a multi-frame case.

### Point time trace + Welch periodogram: `timetrace()` / `periodogram()`

Wall pressure fluctuations at a single point over time, and their
spectrum - `timetrace()` pulls `Data/<surface>/<name>` at ONE raw surfel
across every frame in the file; `periodogram()` wraps
`scipy.signal.welch` on top of that. Works for any variable stored in
the file (pressure, `y+`, forces, ...), not just pressure.

The point is given as `(span_pct, chord_pct)` percentages (0-100) of
`r/R` and `x/c` - the SAME `x/c` convention as `at_radii()`/
`to_common_grid()` (local, per-radius-band, percentile-normalized -
see `chord_percentile`/`reverse_chord` there). The nearest available raw
surfel is used (not an interpolated value) - `_nearest_point()` returns
the requested vs. actual `(r, x/c)`, and every method here reports the
actual location in its `point_info`/plot title, since the nearest surfel
generally won't sit exactly on the target:

```python
t, p, info = sv.timetrace('static_pressure', span_pct=80, chord_pct=90, surface='Upper')
print(info)  # {'idx': ..., 'r': 0.100..., 'xc': 0.901..., 'surface': 'Upper'}

sv.plot_timetrace('static_pressure', span_pct=80, chord_pct=90, surface='Upper',
                   ylabel='Static pressure [Pa]', savepath='p_timetrace.png')

freq, psd, info = sv.periodogram('static_pressure', span_pct=80, chord_pct=90, surface='Upper')
sv.plot_periodogram('static_pressure', span_pct=80, chord_pct=90, surface='Upper',
                     ylabel='PSD [Pa$^2$/Hz]', savepath='p_periodogram.png')
```

**Time axis / sampling rate**: needs a real physical timestep, which
these files don't always have -
`convert_snc_to_h5()` (`converters/ensight_to_h5.py`) writes
`Metadata/mid_s` (real time in seconds) ONLY when an `nc_stats` file was
supplied at conversion time; `SNCReader.to_h5()` never writes any time
info at all. `timetrace()` falls back, in order: explicit `dt` argument
(assumes uniform spacing) -> `Metadata/mid_s` if present -> the raw
integer frame index (fine for just looking at a trace's shape, not a
real time axis). `periodogram()` is stricter - it needs `fs`, `dt`, or a
usable `Metadata/mid_s`; without one of those it raises rather than
silently plotting a meaningless frequency axis.

`nperseg` (Welch segment length) defaults to `min(256, n_frames)`, not
scipy's own bare 256 - with only a handful of frames (a likely case
while more data is still being generated - this whole feature's
motivating use case) scipy would silently clip 256 down to `n_frames`
anyway; doing it explicitly here avoids that surprise.

**Validated** against a synthetic file built with the exact same schema
(`Geometry/Upper,Lower` + `Data/Upper,Lower/<name>` with a frame axis +
`Metadata/lrf_axis_origin,lrf_axis_direction,frame_index,mid_s`) as a
real `to_h5()`/`convert_snc_to_h5()` output, since no real multi-frame
file was available locally: a point's requested `(80%, 90%)` location
resolved to the correct nearby raw surfel (`r/R=0.802`, `x/c=0.901`);
`timetrace()` correctly picked up `Metadata/mid_s` for its time axis
(`dt=0.001` s, matching the synthetic sampling rate) with no `dt`
argument passed; a 50 Hz sine injected into the synthetic signal (plus
noise) came back as a sharp, correctly-located peak at `50.8` Hz (off by
one Welch frequency bin - as expected, not exact) in `periodogram()`'s
output, with a flat noise floor everywhere else.

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
- `SurfaceVariable`'s `reverse_chord` (which end of the raw chord axis is
  the leading vs. trailing edge) has no automatic detection - it's
  arbitrary per case, currently determined by eye (checking whether the
  Kutta-condition/near-zero-Cp end and the stagnation-point/suction-peak
  end land where expected) rather than computed from anything in the
  file. `SurfaceVariable.cp(stat='rms')` is implemented and its reduction
  formula is validated synthetically, but hasn't yet been checked against
  a real multi-frame pressure file - only one real frame was available
  locally when this was built.
