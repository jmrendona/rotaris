# Rotaris

Python based post-procesing tool for rotor simulations whose data was computed from PowerFLOW solver.

## 1. Volumetric planes/cuts → `.fnc` (pf2ens + pyvista, see `converters/fnc_plane.py`)

`.fnc` files are PowerFLOW's fluid measurement with a multi-resolution octree data.
Same rule as pressure above: this goes through `pf2ens` (per frame) +
`pyvista` sampling, never PowerVIZ, never a direct `.fnc` read.

Two extraction "shapes" are implemented so far:

- **`fnc-meridional`**: a single rotated hub-to-tip, inlet-to-outlet plane
  at one fixed azimuth angle.
- **`fnc-iso-radius`**: a fixed-radius cylindrical cut, unrolled into an
  azimuth-vs-axial-position 2D grid.

Both take a `--variables` list of `pf2ens` short codes (run
`pf2ens -d <fnc_path>` against your file to see what's available, e.g.
`vmag,p,vx,vy,vz`) and a `--first`/`--last` frame range.

### Masking the obtained images

**`--freeze-mask-variable vmag`** is the one masking option that is
validated and safe to use. PowerFLOW initialize all the computational
domain (including inside the solids) with the lattice reference values.
These, nertheless, are never updates when located inside
solid, body-fixed geometry. This leaves that across frames their
value barely changes, unlike real flow. This flag computes `Data/frozen`
from that near-zero-variance signature and needs `--first`/`--last` to
span at least 2 (ideally several, spread out) frames to be reliable.
`fnc_plane.plot_frame()` (and `fnc-plot` below) automatically combines
`Data/frozen` into the valid-point mask if it's present.

**There is no working mask for the rotor blades.** `vtkValidPointMask`
(pf2ens/pyvista's own solid-geometry flag) does not track the blades at
all near mid-to-tip span yet. A real, axially-aware blade mask was attempted
(see `RotorBladePosition` in `fnc_plane.py`) and reverted after several 
rounds because it never held up as a trustworthy, full-plot-scale result. 
What *is* validated and available is **identification only**: `RotorBladePosition.blade_azimuths_deg()` that
draws a line at each blade's true instantaneous azimuth. This is useful as a visual
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
`--freeze-mask-variable` to calibrate correctly. See `fnc_plane.py`'s
notes on threshold calibration if the default `--freeze-rel-threshold
0.01` over- or under-masks for your case. `fnc-freeze-mask` recomputes
`Data/frozen` on an existing file post-hoc if you need to retune this
without re-running the `pf2ens` extraction.

## 2. Conversion Forces, Normals, Geometry → `SNCReader` (raw `.snc`)

`converters/snc_reader.py` reads a PowerFLOW surface measurement `.snc`
file directly (NetCDF) and converts it to a compact HDF5.

- `Surface X/Y/Z-Force` and `Skin Friction` are converted to physical units
  (Pa) automatically in `to_h5()`, using the file's own lattice scale
  factors. Validated against the file's own stored `Skin Friction` with a of
  correlation ~0.999.
- Surfel normals, centroids, areas, and rotor axis/RPM
  metadata come along with it.
- `Static Pressure` is deliberately **not** converted here (see below).
  It's written in Lattice Units, flagged `physical_units=False`.

No cluster-side extraction step is needed. The `.snc` file
itself is the input:

```python
from converters.snc_reader import SNCReader

reader = SNCReader("forces_rotor.snc")
reader.to_h5("forces_rotor.h5")
```

### Very large (DNS-resolution) meshes: a 32-bit format limit, fixed

`SNCReader` opens the file via `_LargeRecordNetcdfFile` (a small
`scipy.io.netcdf_file` subclass in the same module), not `scipy.io.netcdf_file`
directly. Plain scipy fails with `ValueError: read length must be
non-negative or -1` once the "measurements" record variable's true
per-record byte size (surfel count x recorded variables x item size -
NOT overall file size, which is dominated by the separately-read, unaffected
fixed-size geometry arrays) crosses what the classic NetCDF format's
32-bit `vsize` header field can hold. Confirmed on a real case: an 8 GB
`.snc` file hit this, a 5.4 GB file from the same project did not - it's
specifically the measurements block's own size, not the file total.

scipy's own source documents the failure mode without handling it: a
writer facing an unrepresentable `vsize` is supposed to store the escape
sentinel `2^32-1` instead, which scipy's *signed* 32-bit parse turns into
`-1`, corrupting the internal record-size accounting (the same
signed/unsigned mismatch already misreads any legitimate size between
~2.1-4.3 GB as negative too, even without hitting the "official" escape
case). The fix: never trust the file's own `vsize` for this - recompute
it independently from the variable's own shape/dtype (unaffected by the
32-bit field), exactly like scipy's own comment describes but never
implements. See `_LargeRecordNetcdfFile`'s docstring for the full
mechanism.

**Validated**: reproduced the exact reported error by surgically
corrupting a real (small) NetCDF file's `vsize` field to the documented
sentinel value - plain `scipy.io.netcdf_file` failed with the identical
error message; `_LargeRecordNetcdfFile` recovered the correct record
size and the exact original data. Also confirmed `_LargeRecordNetcdfFile`
behaves identically to plain scipy on a normal (uncorrupted) file - no
regression.

That fix alone surfaced a **second, separate** 32-bit ceiling on the
actual 8 GB DNS case: `ValueError: invalid shape in fixed-type tuple:
dtype size in bytes must fit into a C int`. Even with `_recsize`
correct, scipy still reads every record variable (any NetCDF-level
array whose first axis is the frame dimension - "measurements" is one;
this format has at least one other, smaller one riding the same frame
axis too, e.g. a per-frame timestamp) through one NumPy *structured*
dtype (one "field" per record variable, each field a fixed-shape
sub-array format string). NumPy's structured-dtype machinery computes
each field's byte size as a C `int` internally and refuses anything
past ~2.1 GB *per field* - unrelated to the NetCDF `vsize` issue above,
and not fixable by correcting `_recsize`, since the failure happens
while NumPy is still building the dtype, before any data is read.

Fix: `_LargeRecordNetcdfFile` never builds a structured dtype for
record variables at all, regardless of how many there are. NetCDF's
classic format interleaves record variables by record (each variable's
data, back-to-back, in declaration order, padded to a 4-byte boundary),
and each variable's own file offset (`begin_`, already correctly parsed
as a genuine 64-bit value) tells us exactly where its data starts - so
the whole interleaved record block is read once as raw bytes, and each
record variable gets its own plain (non-structured) strided view into
it. A plain ndarray's shape/strides are 64-bit, with no such ceiling. An
earlier version of this fix only bypassed the structured dtype when
there was exactly one record variable, which doesn't match this file's
actual layout (it has more than one) and hit the same ceiling through
the leftover fallback path - this version has no such precondition.

**Validated**: reproduced the exact NumPy error directly (constructing
the same shape of structured dtype scipy would have built for the real
8 GB file's variables) and confirmed the described ~2.1 GB per-field
ceiling. Built real multi-record-variable NetCDF files - one mirroring
this file's actual layout (a large float record variable alongside a
small one sharing the frame axis), and one exercising the byte-alignment
padding case (an odd-sized int16 record variable interleaved with a
float one) - and confirmed `_LargeRecordNetcdfFile` reads every variable
back byte-for-byte identical to plain scipy in both cases, and still
matches on the original single-record-variable file too - no
regression. Not yet run end-to-end against the actual multi-GB `.snc`
file (needs the HPC) - validated at the exact mechanism level.

## 3. Static Pressure → `pf2ens` (do not derive it from raw `.snc`)

PowerFLOW translates static pressure between lattice and real units through
an internal Cp-based mechanism that could not be reverse-engineered from
the file's own stored scale factors. `pf2ens`
does this conversion correctly and is the only source trusted for pressure.

### Running this on the cluster

**You do not run `pf2ens` yourself.** `convert_snc_to_h5()` (in
`converters/ensight_to_h5.py`) calls it for you internally, once per
frame, and deletes each frame's intermediate EnSight export (~800 MB)
before moving to the next one, so nothing piles up on disk. All you run
is this one Python function (or the equivalent CLI command).

**Prerequisites**: `pf2ens` must be on `$PATH`. It already is inside
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
  range to convert. It has to match frames what actually exist in the
  file. Check with `SNCReader(snc_path).n_frames` if unsure or with 
  `exafile` build-in tool from PowerFLOW.
- `--surface-split`: optional. Splits into `Upper`/`Lower` groups. Opens `snc_path` a    
second time internally to borrow its
  validated classification. No extra input file needed.
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
python convert.py forces <snc_path> <output.h5> [--face-name NAME] [--surface-split]
python convert.py pressure <snc_path> <output.h5> --first N --last M [--surface-split] [--nc-stats FILE] [--reference-frame N] [--work-dir DIR]
python convert.py fnc-meridional <fnc_path> <output.h5> --angle DEG --variables v1,v2 --first N --last M [--freeze-mask-variable vmag]
python convert.py fnc-iso-radius <fnc_path> <output.h5> --radius M --variables v1,v2 --first N --last M [--freeze-mask-variable vmag]
```

`run_conversion.sh` submits either subcommand as a single-node SLURM batch
job. Nothing here is parallelized, so it only ever requests one node/one
task:

```bash
sbatch run_conversion.sh forces /path/case.snc /path/out.h5 --surface-split
sbatch run_conversion.sh pressure /path/case.snc /path/out.h5 --first 0 --last 199 --surface-split
sbatch run_conversion.sh fnc-meridional /path/case.fnc /path/plane_0deg.h5 \
    --angle 0 --variables vmag,p --first 0 --last 99 --freeze-mask-variable vmag
sbatch run_conversion.sh fnc-iso-radius /path/case.fnc /path/iso_r85pct.h5 \
    --radius 0.274 --variables vmag,p --first 0 --last 99 --freeze-mask-variable vmag
```

**Use `sbatch`, not a direct shell command, for any `fnc-*` extraction**.
`.fnc` files are far larger than `.snc` making it not something suitable 
to run on a login node. `fnc-plot` and `fnc-freeze-mask` are the
only exceptions worth running directly in a shell as they only touch an 
already-extracted, comparatively small `.h5` file, no `pf2ens`/`.fnc` access at all.

Everything after `run_conversion.sh` on the command line is forwarded
straight to `convert.py`.

### One-time setup, before the first `sbatch` submission: 

Alliance compute nodes have no internet access, so the Python venv `run_conversion.sh`
expects (`~/rotaris-venv`) has to be built from a **login node** first:

```bash
module load StdEnv/2023 python/3.11 scipy-stack/2023b vtk
virtualenv --no-download --system-site-packages ~/rotaris-venv
source ~/rotaris-venv/bin/activate
pip install --no-index --upgrade pip
pip install pyvista
```

`run_conversion.sh` will refuse to run (with a clear message) if `~/rotaris-venv`
doesn't exist yet, rather than trying to build it inside the batch job.
After that, `run_conversion.sh` just reuses the venv.

**Where things live**: `run_conversion.sh` locates `convert.py` via
`ROTARIS_DIR` (default `$HOME/rotaris`), NOT via its own script path -
`sbatch` copies the submitted script to a per-job spool directory on the
compute node (`/var/spool/slurm/slurmd/job<ID>/`) and runs that copy, so
`${BASH_SOURCE[0]}`'s directory would resolve to the spool path, not
wherever `rotaris/` actually lives (confirmed: this is exactly what
produced `python: can't open file '/var/spool/.../convert.py'` before
this was fixed). If `rotaris/` isn't at `$HOME/rotaris`, override it:
`ROTARIS_DIR=/path/to/rotaris sbatch run_conversion.sh ...`.

You can `sbatch ~/rotaris/run_conversion.sh ...` from wherever you want
the job's data/logs to land (typically `$SCRATCH` or your `/project`
allocation) - no need to `cd` into the code folder first. SLURM's
`--output`/`--error` logs land in whatever directory you ran `sbatch`
from, and `<snc_path>`/`<output>` are resolved relative to that same
directory - **`cd` into that target directory before running `sbatch`**
if it's somewhere writable from compute nodes (e.g. `$SCRATCH`); some
clusters (e.g. Trillium) make `$HOME` read-only from compute nodes, so
submitting from `$HOME` with the default relative `--output`/`--error`
fails with a read-only-filesystem error.

`--time`/`--mem`/`--cpus-per-task` at the top of `run_conversion.sh` are
placeholder that need to be adjusted before running depending on the case.

## Splitting into upper/lower surface

Manage the division between suction and pressure side of the interest geometry.
Needed for anything that requires knowing which side of the blade a point
is on (e.g. skin-friction lines).

- **`SNCReader.surface_split()`**: uses the sign of the surfel normal's component
  along the rotation axis (`lrf_axis_direction`). This is the
  trusted classification used directly for raw-`.snc`-derived data
  (`to_h5(..., surface_split=True)` writes separate `Geometry/Upper`,
  `Geometry/Lower`, `Data/Upper`, `Data/Lower` groups).
- **`EnsightFrame.surface_split()`**: do **not** use the geometry from
  pf2ens's own mesh as `pf2ens` splits complex surfels into quads/trias
  for EnSight compatibility, which breaks mesh connectivity enough that
  VTK's `compute_normals()` becomes locally inconsistent. Instead, 
  classification is borrowed from the raw `.snc` file via nearest-neighbor 
  position matching:

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
  the mesh's own bounding-box midpoint, not on `lrf_axis_origin`. Skipping 
  this re-centering silently breaks the nearest-neighbor match on the 
  axis-aligned coordinate. As the two only coincide (by symmetry) for the 
  axes perpendicular to the rotation axis.

  `convert_snc_to_h5(..., surface_split=True)` (and `--surface-split` on
  the CLI) already handles all of this automatically.

## Output file structure

### `SNCReader.to_h5()` - all frames in the file, one `.h5` file

Geometry (positions/normals/areas) is written once. Every
variable is written as a 2D dataset covering every frame in the file:

```
Metadata/              lrf_axis_origin (meters), lrf_axis_direction, frame_index
                       (datasets, frame_index = arange(n_frames)); scale_*,
                       offset_*, lrf_angular_vel_lattice (attrs)
Geometry/X,Y,Z         surfel centroid positions, meters (1D, shape (n_points,))
Geometry/Normal_X,Normal_Y,Normal_Z, Area   normals (unit vectors) and area, m^2
Data/<Variable_Name>   one dataset per .snc variable (spaces -> underscores),
                       shape (n_frames, n_points), each with attrs
                       lattice_unit_class, physical_units
```

With `surface_split=True`, `Geometry` and `Data` are each replaced by
`Geometry/Upper`, `Geometry/Lower`, `Data/Upper`, `Data/Lower`. `n_frames` is read
directly from the file (`reader.n_frames`, from `measurements`' first
axis). `to_h5()` always writes every frame present without the need of an additional
argument.

### `EnsightSeriesWriter` / `convert_snc_to_h5()` - multi-frame

One `.h5` file total, not one per frame and not one group per frame.
Frames are rows in shared 2D datasets:

```
Geometry/X,Y,Z                  written ONCE, from the is_reference=True
                                frame (1D, shape (n_points,)). No
                                normals (see below)
Data/<variable>                 one 2D dataset per variable,
                                shape (n_frames, n_points). Each
                                add_frame() call appends a row
Metadata/frame_index, start_ts, end_ts, mid_ts, mid_s, lrf_position_rad
                                1D, shape (n_frames,). Row i describes
                                Data's row i
Metadata/lrf_axis_origin, lrf_axis_direction
                                frame-independent (rotation axis doesn't
                                change), meters. Matches SNCReader.to_h5()'s
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
`pf2ens` derived variables, which don't need
normals. `EnsightFrame.normals()` isn't reliable enough to propagate
downstream for anything else  (see the section above and
`EnsightFrame.normals()`'s docstring). Upper/lower classification always
comes from `SNCReader.surface_split()` (raw `.snc` normals) instead.

Storing geometry once assumes the blade is rigid (only orientation
changes between frames, not shape). Noted as an assumption in
`EnsightSeriesWriter`'s docstring, with per-frame geometry storage
mentioned as the more foolproof (but more storage consuming) alternative,
not implemented.

## Wall shear / friction lines: `bladeprocessor/FrictionLines` - Equations

`FrictionLines` consumes a `SNCReader.to_h5(..., surface_split=True)` file
(the forces branch above, never the `pf2ens`/pressure branch, whose
recomputed normals aren't trustworthy for this, see "Splitting into
upper/lower surface"). Every quantity below is per-surfel unless noted.

**Wall shear vector** (`wall_shear()`) - the surface force with its
normal (pressure) component removed, leaving only the tangential
(friction) part:

```
tau = F - (F . n) n
```

`F` is `Surface X/Y/Z-Force` (already Pa, from `SNCReader.to_h5()`), `n`
is this surfel's own unit normal (`Geometry/Normal_X,Normal_Y,Normal_Z`, from
the raw `.snc`).

**Skin friction coefficient** (`cf()`) - normalized by the local dynamic
pressure, using each surfel's own radius:

```
Cf = tau / q_ref
q_ref = 0.5 * rho_ref * (omega * r)^2
omega = rpm * 2*pi / 60
```

`r` is this surfel's physical radius from the rotor's rotation axis
(`lrf_axis_origin`/`lrf_axis_direction`, the one place `FrictionLines`
still uses the rotation axis rather than raw Cartesian position, see
`_radius()`). This matches `BladePostProcessor.compute_cf()` elsewhere in
this project (`U_ref(r) = omega * r`), rather than
non-dimensionalizing by one global freestream/tip velocity. **Caveat**:
`q_ref -> 0` as `r -> 0`, so `Cf` blows up near the rotation axis. Some
faces include a hub geometry right at `r~0` if no entities are defined. 
If that is the case it can be excluded (`span_min`/`span_max` in
`friction_lines()`, or just don't query `cf_at_radii()` near `r=0`)
before trusting an unrestricted `cf()` call.

`component` picks which part of `tau` to report:

```
Cf (magnitude)  = |tau| / q_ref                    >= 0 always
Cf (chordwise)  = tau[chord_axis] / q_ref           signed
Cf (spanwise)   = tau[span_axis]  / q_ref           signed
```

Magnitude just dips towards zero but can never show a sign reversal
(separation/reattachment). The signed components can, which is the
point of having them (see `friction_lines_test_cf_chordwise_*.png`
crossing zero mid-chord at the outer radii).

**Local chordwise position** (`cf_at_radii()`): for a thin band of
surfels around a target radius (`|r - r_target| < tol`), the raw
Cartesian chord coordinate is rescaled to `[0, 1]` using that band's own min/max:

```
x/c = (chord - chord_min) / (chord_max - chord_min)
```

This is a per-band, per-case rescaling, not a case-independent x/c (see
the class docstring's caveat and README's "What's still open" below).
The resulting `(x/c, Cf)` pairs are then averaged into `n_chord_bins`
equal-width x/c bins  to obtain a mean Cf per bin. Without this, a single radius
band on a real `.snc` surface contains hundreds of thousands of raw,
noisy points, which reads as a dense cloud rather than a curve once
plotted. The point of `n_chord_bins` is turning that cloud into the
single readable curve per radius that `plot_cf_radii()` draws.

If all the geometry is lumped into a single `/Rotor::Default-Segment` face,
and therefore each blade contribution can not be separated, the option
`span_min`/`span_max` needs to be activated so a radius band picks up
only the desired blade. Otherwise, both blades surfels will be used at
once generating a spurious extra peak at both x/c=0 and x/c=1 instead of 
one, near the real leading edge (stacking the two blades results together).
It is the same option as  `span_min`/`span_max` cropping `friction_lines()` 
has implemented.

Similarly, as the `span_min`/`span_max` divides the geometry, it can happen that
the band crosses the blades TE before passing the LE, thus generating an
inverse in the peak location. Therefore, the option `reverse_chord` parameter
is added here. A similar ambiguity is also documented for Cp plots.

### Cf unsteadiness: `cf(stat=...)` / `cf_at_radii()`/`plot_cf_radii(stat=...)` / `friction_lines(stat=...)`

`stat='rms'` (also `'raw_rms'`) on `cf()` - and forwarded from
`cf_at_radii()`/`plot_cf_radii()`/`friction_lines()` - gives the actual
"how unsteady is this" statistic: the RMS fluctuation of Cf about its
mean, at every surfel. This flags things the MEAN Cf field alone can
miss entirely - transition, an unsteady separation/reattachment line
whose position wanders in time rather than sitting still, or a vortex
core (see `critical_points()`'s `'focus'` points) that moves around.

Computed on the SCALAR Cf itself (magnitude or a signed component),
frame by frame, THEN reduced - not by reducing the wall-shear VECTOR
first and taking its magnitude/component afterward. Those differ:
extracting a magnitude/component is a nonlinear operation, so
`RMS(|tau|) != |RMS(tau)|` in general - unlike the MEAN, which commutes
with the linear `tau = F - (F.n)n` projection (that's why
`wall_shear(frame=None)`'s existing mean behavior needed no change to
support this).

```python
cf_rms = fl.cf(surface='Upper', frame=None, component=None, stat='rms')

fl.plot_cf_radii([0.045, 0.072, 0.1], surface='Upper', stat='rms', span_min=0.02,
                  reverse_chord=True, savepath='cf_rms_radii.png')

# spatial map - the quiver direction always stays MEAN flow regardless of
# stat (an "RMS direction" isn't a meaningful vector):
fl.friction_lines(surface='Upper', stat='rms', span_min=0.02, savepath='cf_rms_map.png')
```

**Validated** against a synthetic case with a known analytic answer (a
uniform oscillating chordwise force, `A + B*cos(2*pi*f*t)` with `A > B`
so magnitude and the chordwise component are identical throughout, no
sign flip): both `component=None` and `component='chordwise'` recovered
`Cf_rms = (B/sqrt(2)) / q_ref(r)` to within floating-point precision
(~1e-15 relative error) at every point; the mean matched `A / q_ref(r)`
exactly too; the existing `frame=<int>` (instantaneous) path was
unaffected (checked against its pre-change values) - the whole `stat`
addition is a pure extension, not a behavior change to what already
worked.

### Separation/reattachment line: `separation_line()` / `plot_separation_line()` / `save_separation_line()`

It is based on every span location where chordwise Cf (`tau[chord_axis] / q_ref`)
crosses zero. Restricted to ONE blade section via `span_min`/`span_max`, same reason as
`cf_at_radii()`/`friction_lines()`.

**Method:** partitioning the (cropped) span range into `n_span_bins` bins
(default 200). Within each, chordwise Cf is averaged into `n_chord_bins` x/c bins (default 200)
using the SAME per-band, percentile-normalized x/c convention as
`cf_at_radii()` (pass the same `reverse_chord` used in
`plot_cf_radii()`, or it might disagree on which end is the leading
edge). Adjacent bins with a sign change are linearly interpolated to
localize each crossing. A span bin can produce zero, one, or several
crossings; all are kept.

**Labeling is NOT based on the raw sign of Cf.** Which physical position
counts as separation and reattachment is defined by an ordering method. The crossings within
each span bin are labeled purely by order along x/c: the 1st, 3rd, 5th,
... is `'separation'` (entering a reversed-flow region), the 2nd, 4th,
6th, ... is `'reattachment'` (leaving it) always alternating, always
paired, regardless of which raw sign happens to mean "attached" on this
mesh. Each matched pair shares a `pair_id` (a bin with an odd number of
crossings leaves one unpaired, `pair_id=-1` - the region it opened
extends past the resolved x/c range rather than closing within it, e.g.
all the way to the trailing edge).

```python
points = fl.separation_line(surface='Upper', frame=None, span_min=0.02, reverse_chord=True)
# [{'span': ..., 'chord': ..., 'r': ..., 'xc': ..., 'kind': 'separation'|'reattachment', 'pair_id': ...}, ...]

fl.save_separation_line(points, 'separation_line.txt')  # span_m, chord_m, r_m, xc, kind, pair_id - tab-separated

# overlay on friction_lines() directly:
fl.friction_lines(surface='Upper', frame=None, span_min=0.02, show_separation_line=True,
                   separation_line_kwargs={'reverse_chord': True}, savepath='friction_lines_sep.png')

# or on any existing (span, chord) Axes:
fl.plot_separation_line(ax, points)
```

`plot_separation_line()` draws large, black-edged, high-contrast markers
(separation in red, reattachment in cyan), sized to stay visible against
`friction_lines()`'s dense background scatter. `connect_pairs=True`
optionally draws a line between each matched pair marking the reversed
flow region's chordwise extent at that span station. This option is off by default,
since on a dense case the connecting lines packed side by side read as a
solid black wall rather than individually legible segments.

### Spanwise migration-reversal line: `migration_line()` / `plot_migration_line()` / `save_migration_line()`

Every location where SPANWISE Cf (`tau[span_axis] / q_ref`) crosses zero
showing where near-wall flow switches between migrating toward the tip
("outward", e.g. classic centrifugal pumping in a rotating boundary
layer) and migrating toward the root ("inward"). This is a different
physical phenomenon from separation/reattachment. Same per-span-band binning architecture and
`span_min`/`span_max`/`reverse_chord` requirements as `separation_line()`. An independent method called `migration_line()` is implemented for this case.

**Labeling is physically grounded, unlike `separation_line()`'s.**
Chordwise Cf's sign has no fixed physical meaning (it depends on this
mesh's arbitrary `chord_axis` orientation - see `reverse_chord` above),
but spanwise Cf's sign can be: span is used as-is, the raw absolute
Cartesian coordinate, and `span_min`/`span_max` already isolates one
blade running from the hub outward. So on that selected half, increasing
span consistently means "toward the tip". A crossing from positive to
negative spanwise Cf is labeled `'inward'`; negative to positive is
`'outward'` unless this mesh's span_axis happens to point the opposite
way on whichever half was cropped, in which case `flip_direction=True`
swaps the meaning.

```python
points = fl.migration_line(surface='Upper', frame=None, span_min=0.02, reverse_chord=True)
fl.save_migration_line(points, 'migration_line.txt')  # span_m, chord_m, r_m, xc, kind, pair_id

fl.friction_lines(surface='Upper', frame=None, span_min=0.02, show_migration_line=True,
                   migration_line_kwargs={'reverse_chord': True}, savepath='friction_lines_mig.png')
```

### Vortex-footprint critical points: `critical_points()` / `plot_critical_points()` / `save_critical_points()`

Locations where the complete wall-shear vector (chordwise Cf, spanwise Cf)
vanishes simultaneously.
Per Lighthill's theorem (see Tobak & Peake, 1982, "Topology of
Three-Dimensional Separated Flows"), any point a real 3D flow's surface
streamlines converge to, diverge from, or spiral around must be a point
of zero skin friction, so genuine 3D structures, including vortex
footprints specifically, can be located this way.

Unlike `separation_line()`/`migration_line()` (which bin per span band
using a locally renormalized x/c), this bins the raw surfel cloud onto a
genuine 2D grid in absolute physical `(span, chord)` coordinates [m]. A
real 2D neighborhood search needs a consistent coordinate system across
neighboring cells. No `reverse_chord` parameter as a result. Raw
physical chord already has a real, consistent geometric meaning on its own.

Method: bins the cropped selection onto an `n_span_bins x n_chord_bins`
grid (mean chordwise/spanwise Cf and Cf magnitude per cell, cells with
fewer than `min_count` surfels treated as unreliable); candidates are
cells whose Cf magnitude is both in the bottom `magnitude_percentile`%
of the grid and a local minimum among their neighbors; each
candidate is classified via the local Jacobian of `(chordwise Cf,
spanwise Cf)` w.r.t. `(chord, span)` (central finite differences), using
its eigenvalues:
- `'node'` - real eigenvalues, same sign - lines converge to/diverge
  from this point (a 3D separation/attachment node)
- `'saddle'` - real eigenvalues, opposite sign - lines pass through/
  around it (a typical reattachment saddle)
- `'focus'` - complex eigenvalues - lines SPIRAL around it - the actual
  footprint of a vortex core (LEV, corner/horseshoe vortex, ...), not
  just an ordinary separation/reattachment feature

```python
points = fl.critical_points(surface='Upper', frame=None, span_min=0.02)
fl.save_critical_points(points, 'critical_points.txt')  # span_m, chord_m, r_m, cf_mag, kind

fl.friction_lines(surface='Upper', frame=None, span_min=0.02, show_critical_points=True,
                   savepath='friction_lines_crit.png')
```

#### Theory: why eigenvalues tell you node vs. saddle vs. focus

A surface streamline is an integral curve of the wall-shear
vector field: parametrize the curve by an arclength-like variable `t`,
and it satisfies

```
d(chord)/dt = u(chord, span)      where  u = Cf_chordwise
d(span)/dt  = v(chord, span)      where  v = Cf_spanwise
```

`t` here isn't physical time, it's just how far you've walked along the
surface following the local wall-shear direction. A critical point
`(chord_0, span_0)` is where `u = v = 0` simultaneously: the direction
field is undefined there, which is exactly why real streamlines converge to it, diverge from it, or
spiral around it, instead of just passing through like everywhere else.

Near such a point, Taylor-expand `(u, v)` to first order (the
higher-order terms vanish fastest as you approach the point, so the
linear part determines the local picture):

```
[u]   [du/dchord  du/dspan] [chord - chord_0]
[v] ~ [dv/dchord  dv/dspan] [span  - span_0 ]  =  J * (x - x_0)
```

`J` (the 2x2 Jacobian `critical_points()` estimates by central finite
differences on the grid) turns the messy nonlinear flow near the point
into a simple linear system, `dx/dt = J x` (relative to `x_0`) - and the
solutions of a linear system like this are completely characterized by
`J`'s eigenvalues `lambda_1, lambda_2` (roots of
`lambda^2 - tr(J) lambda + det(J) = 0`, where `tr(J)` is the trace and
`det(J)` the determinant):

| `tr(J)^2 - 4 det(J)` | eigenvalues | pattern | `kind` |
|---|---|---|---|
| `> 0`, `det(J) > 0` | real, same sign | every line converges to (`tr(J)<0`) or diverges from (`tr(J)>0`) the point | `'node'` |
| `> 0`, `det(J) < 0` | real, opposite sign | lines converge along one direction, diverge along the other - only 2 lines actually touch the point, everything else is deflected around it | `'saddle'` |
| `< 0` | complex conjugate pair, `lambda = a +/- bi` | the `bi` part is a ROTATION - lines spiral in (`a<0`) or out (`a>0`) around the point | `'focus'` |

Physically:
- A **node** is where surface flow genuinely collects (a 3D
  reattachment point flow spreads out from, or an attachment point it
  converges to). No rotation, just pure convergence/divergence.
- A **saddle** is the generic "flow gets redirected around an obstacle"
  point. Most reattachment lines (as opposed to points) are built from
  chains of these.
- A **focus** is the only one of the three with genuine rotation baked
  into its linearization (`b != 0`, a nonzero imaginary part) - which is
  precisely the mathematical signature of a vortex: the near-wall flow
  doesn't just converge or get deflected, it winds around the point.
  This is why `'focus'`, not `'node'` or `'saddle'`, is the marker
  that actually means "vortex footprint" (leading-edge vortex, corner/
  horseshoe vortex, ...) rather than an ordinary separation/reattachment
  feature.

This eigenvalue classification is the standard tool for this kind of
analysis. See Tobak & Peake (1982), "Topology of Three-Dimensional
Separated Flows" already cited above, or Poincaré's original
classification of singular points of planar vector fields, which this
is a direct application of.

#### Theory: the Poincaré-Hopf check - `poincare_index()`

Each critical point carries an "index": +1 for a `'node'` or a `'focus'`
(the vector field's direction winds around the point once, in the same
sense you walk around it), -1 for a `'saddle'` (winds around once in the
opposite sense). `poincare_index(points)` sums these:

```
index = N + F - S
```

(`N`, `F`, `S` = counts of node/focus/saddle points). The **Poincaré-Hopf
theorem** says that for a vector field on a closed surface (no boundary
- e.g. a full blade's entire skin, both surfaces and the tip cap,
stitched into one topological sphere), this sum must equal the surface's
Euler characteristic `chi` - `2` for anything sphere-like, the
same `2` in `V - E + F = 2` for a polyhedron. It's a real constraint: no
matter how complicated the surface flow pattern looks, the critical
points occurring on a closed surface can't be arranged arbitrarily,
their indices are forced to sum to `chi`.

**This project's `critical_points()` runs on an open patch**, one
surface (Upper or Lower), further cropped by `span_min`/`span_max`, not
a closed surface, so there is no reason `N + F - S` should come out to
`2` here, and it usually won't (this project's own case: `N=10, F=14,
S=10`, `index = 14`). `show_critical_points_index=True` on
`friction_lines()` (or calling `poincare_index()` directly) reports the
number as a text box on the figure regardless, annotated `"(closed-
surface value)"` only in the special case it happens to equal 2. It's a
genuine diagnostic (e.g. a useful self-consistency check across a
resolution change. If the same underlying flow gives a wildly different
index at a different `n_span_bins`/`n_chord_bins`, that's a sign the grid
is under-resolving something, not that the flow changed), not a
pass/fail test on this kind of open selection.

```python
crit_points = fl.critical_points(surface='Upper', frame=None, span_min=0.02)
fl.poincare_index(crit_points)  # -> 14 on this project's own case

fl.friction_lines(surface='Upper', frame=None, span_min=0.02,
                   show_critical_points=True, show_critical_points_index=True,
                   savepath='friction_lines_crit_index.png')
```

This is a real but approximate, resolution- and noise-sensitive tool. Treat results as candidates to inspect against `friction_lines()`'s own
quiver pattern, not as ground truth by themselves.

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
axis-based radius) are identical to `FrictionLines`, it assumes a little blade twist.

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

**Instantaneous vs. average vs. RMS:** every method takes the same
`frame`/`stat` pair `FrictionLines` uses: `frame=<int>` for one frame
directly (`stat` ignored as a statistic across frames isn't meaningful for
a single one), `frame=None` (default) reduces across every frame via
`stat` (`'mean'`, `'rms'` = fluctuation std about the mean, or
`'raw_rms'` = `sqrt(mean(x^2))`, includes any nonzero mean.

As for `FrictionLines.friction_lines()`, this tool also allows for the definition of `span_min`/`span_max` to isolate the effect of a single blade. Similarly, the `reverse_chord=True` is also available for a meaningfull plot.

### Whole-blade surface plot: `plot_variable_surface()`

Generalizes `FrictionLines.friction_lines()` beyond Cf to any scalar
field (static pressure, Cp, `y+`, ...). It has the same style (dense scatter, not
interpolated onto a grid), one row
per surface, colors clipped to a percentile range (two-sided here, since
a general field like Cp can be negative, unlike Cf magnitude):

```python
sv.plot_variable_surface(
    lambda s: -sv.cp(surface=s, stat='mean'),  # note the negation, see "Sign" above
    cbar_label='-Cp', span_min=0.03,
    savepath='cp_surface.png',
)
```

An optional `get_vector` callable overlays a direction quiver (e.g. for a
surface velocity field), same as `friction_lines()`'s wall-shear quiver.

### Cross-case comparison: `to_common_grid()` / `field()` / `SurfaceVariableField`

Resamples any per-surfel field onto a shared `(r/R, x/c)` grid. Built directly from a raw surfel
cloud instead of a pre-resampled `(Radius, Chord)` file. Only meaningful
for cases known to share the same geometry, or ones that are properly
scalable in both span and chord.

The option of performing comparison/delta plotting is also available by using 
`SurfaceFieldComparator` where the plots are made together with the delta using
the first one used as a reference case.

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

`c_ref`, reference chord, (unlike `SurfaceField`, which can fall back to `max(chord)` 
from its own file) must be passed explicitly. Raw surfel data has no native
chord axis to infer one from, and using the same physical `c_ref` across
every case being compared is what keeps `x/c` meaning the same thing in
each.

### Pressure fluctuation: `pressure_fluctuation()` / `plot_pressure_fluctuation()`

`p'(frame) = p(frame) - p_mean`, where `p_mean` is the mean over every
frame in the file. It is the same mean `variable(stat='rms')` uses internally
for `Prms = sqrt(mean(p'^2))`, so this fluctuation is consistent with
that statistic rather than some other baseline. The results are giving in
dimensional units [Pa], not normalized by `q_ref` (unlike `cp()`).

```python
# one frame's fluctuation as a blade contour:
sv.plot_pressure_fluctuation(frame=0, span_min=0.03, savepath='p_fluct_frame0.png')

# one image per frame, e.g. for an animation:
for frame in range(sv.n_frames):
    sv.plot_pressure_fluctuation(frame, span_min=0.03, savepath=f'p_fluct_frame{frame:03d}.png')
```

`Prms` itself needs no new code as it's already settle inside
`variable('static_pressure', stat='rms')`
(or `cp(stat='rms')` for the normalized version), usable directly with
`plot_variable_surface()`/`plot_at_radii()`.

### Point time trace + Welch periodogram: `timetrace()` / `periodogram()`

Wall pressure fluctuations at a single point over time, and their
spectrum. The `timetrace()` option pulls `Data/<surface>/<name>` at one raw surfel
across every frame in the file. The `periodogram()` wraps
`scipy.signal.welch` on top of that. Works for any variable stored in
the file (pressure, `y+`, forces, ...), not just pressure.

The point is given as `(span_pct, chord_pct)` percentages (0-100) of
`r/R` and `x/c`. The same `x/c` convention as `at_radii()`/
`to_common_grid()` (local, per-radius-band, percentile-normalized). 
The nearest available raw surfel is used where - `_nearest_point()` returns
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
`Metadata/mid_s` (real time in seconds) only when an `nc_stats` file was
supplied at conversion time; `SNCReader.to_h5()` never writes any time
info at all. `timetrace()` falls back, in order: explicit `dt` argument
(assumes uniform spacing) -> `Metadata/mid_s` if present -> the raw
integer frame index (fine for just looking at a trace's shape, not a
real time axis). `periodogram()` is stricter as it needs `fs`, `dt`, or a
usable `Metadata/mid_s`; without one of those it raises rather than
silently plotting a meaningless frequency axis.

## Strip forces (Hanson's method input): `bladeprocessor/StripForces`

Per-radial-strip, time-resolved axial/radial/tangential force. These data
are the raw input Hanson's tonal noise method needs (harmonics of
unsteady sectional loading), computed directly from a `SNCReader.to_h5()` 
conversion.

Per-surfel force = `Surface_X/Y/Z-Force` (Pa) x `Area` (m²), projected
onto a physical basis computed at every surfel (not once for the whole
strip, since the radial direction itself varies across a strip's own
surfels): `axial` = the rotation axis direction (constant), `radial` =
unit vector from the axis to the surfel (always points
outward), `tangential` = `axial x radial`. `axial`'s own sign and
`tangential`'s rotation sense both depend on which way this file's own
`lrf_axis_direction` happens to point. An arbitrary modeling choice,
same kind of ambiguity as `reverse_chord` elsewhere so check the sign
against what you expect physically and use `flip_axial`/
`flip_tangential` if it comes out backwards.

`n_chord_bins` optionally subdivides each radial strip further, along
the chord, for cases where a single strip's whole chord can no longer be
treated as one compact acoustic source (a high blade-passing frequency,
or a high enough harmonic in Hanson's method itself). Off by default and
one strip per radial band is used.

```python
from bladeprocessor.strip_forces import StripForces

sf = StripForces('forces_rotor.h5', r_tip=0.125)

result = sf.compute(span_min=0.02, n_span_bins=20)  # span limits for one blade analysis
sf.save(result, 'strip_forces.h5', dt=0.000056)      # independent file, reusable without this class again

sf.plot_bar_forces(result, show_totals=True, savepath='strip_forces_bar.png')  # bar + cumulative, one blade only

# chordwise-subdivided (non-compact chord case):
result_2d = sf.compute(span_min=0.02, n_span_bins=20, n_chord_bins=5)
```

`plot_bar_forces()`'s x-axis is `r/R` by default (`normalize_radius=True`,
needs `r_tip`). You can pass `normalize_radius=False` for physical 
radius [m] instead.

### Integrated totals: `total_loads()` / `compute()`'s `'totals'` key

Net thrust, torque, and (for completeness) net radial and tangential
force, summed directly over every selected surfel. It is independent of
strip binning entirely, so it's the right quantity to cross-check
against an independently known total (a `.csnc` file output of `exaritool forces.ri`), not a re-added sum of `compute()`'s strips (though
they agree, since strip binning is an exact partition of the same
surfels).

- `'thrust'` = `Sum(F_axial)`.
- `'torque'` = `Sum(F_tangential_i * radius_i)` - a moment, not the same
  as `'tangential_force'`; needs each surfel's own radius as
  a lever arm, not just the raw force.
- `'radial_force'` = `Sum(F_radial)` - usually small; most of a real
  blade's outward pull is centrifugal. If it's not small relative to
  thrust, worth a second look (Hanson assumption).
- `'tangential_force'` = `Sum(F_tangential)` - the net in-plane force,
  related to (but distinct from) the classic rotorcraft in-plane
  "H-force".

`compute()` computes these from the same surfel selection used for
its own strip breakdown and embeds them as `result['totals']`. It has a
guaranteed consistentcy with whatever `span_min`/`span_max` that
particular `compute()` call used. `total_loads()` is the same
computation as a standalone call without the strip processing.

```python
result = sf.compute(span_min=0.02, n_span_bins=20)
result['totals']['thrust']   # (n_frames,) - matches total_loads(span_min=0.02) exactly

totals = sf.total_loads(span_min=0.02)  # standalone, same numbers, no strip binning needed

sf.plot_bar_forces(result, show_totals=True, savepath='strip_forces_bar.png')  # annotates the figure
```

### Thrust/torque coefficients: `plot_bar_forces(rho=..., n_rot=..., diameter=...)`

If `rho` [kg/m^3], `n_rot` [rev/s - not rad/s], and `diameter` [m] are
all given, `plot_bar_forces()`'s y-axis (bars and the cumulative curve)
becomes a standard propeller-convention force coefficient, the same
equation applied to all three components:

```
C_T,axial      = thrust           / (rho * n_rot^2 * diameter^4)
C_T,radial     = radial_force     / (rho * n_rot^2 * diameter^4)
C_T,tangential = tangential_force / (rho * n_rot^2 * diameter^4)
```

This is deliberately not a separate "torque coefficient" for the
tangential bar: what's plotted there is still a per-strip/cumulative
force, and torque isn't a force (see `total_loads()`'s note on why
torque needs a radius-weighted sum). The proper torque coefficient, 
computed from `total_loads()`'s radius-weighted torque (not the 
tangential bar), needs one extra factor of `diameter`:

```
C_Q = torque / (rho * n_rot^2 * diameter^5)
```

Both `C_T,*` and `C_Q` only ever appear in the `show_totals` box, never
as their own bar/line.

**`show_totals` becomes coefficients-only in this mode** when
`rho`/`n_rot`/`diameter` are given, the box shows `C_T,axial`/
`C_T,radial`/`C_T,tangential`/`C_Q` and nothing else; the dimensional
N/N.m totals it would otherwise show are deliberately left out, for
cases where the raw loads shouldn't be exposed but their non-dimensional
form is fine to share.

```python
sf.plot_bar_forces(
    result, show_totals=True,
    rho=1.22523, n_rot=6000 / 60, diameter=0.25,  # n_rot in rev/s, not RPM
    savepath='strip_forces_bar_coeffs.png',
)
```

### Average vs. instantaneous cases

`compute()`/`total_loads()`/`plot_bar_forces()` work on either an
already time-averaged ("average", `n_frames=1`) or a transient
("inst", `n_frames>1`) file. `frame=None` averages over whatever
frames are present (a no-op on a 1-frame file), `frame=<int>` picks one.
`plot_time_trace()`/`phase_lock()`/`harmonics()` below only make sense
for an "inst" file (there's nothing to trace/fold/decompose in a single
already-averaged frame). A clear error is raises if given a
1-frame file rather than silently producing something meaningless.

### Time trace: `plot_time_trace()`

Raw per-strip force vs time, one line per strip. The time-domain view
of `compute()`'s per-frame result.

```python
result = sf.compute(span_min=0.02, n_span_bins=20)  # needs an "inst" (multi-frame) file
sf.plot_time_trace(result, dt=0.000056, component='axial', strips=[0, 4, 9, 14, 19],
                    savepath='strip_time_trace.png')
```

`strips` (0-based indices into `result['radius']`) lets you pick a
legible subset. A real case can have far more strips than fit on one
legend.

### Phase-locked (revolution-folded) forces: `phase_lock()` / `plot_vs_angle()`

Bins every frame by its rotor azimuth angle (needs `rpm`, set in
`__init__`) into `n_azimuth_bins` bins spanning one revolution, and
averages every frame landing in each bin using as many revolutions
as it spans. This is the standard "phase-locked averaging" that turns a
noisy, multi-revolution time series into one clean once-per-rev curve.

```python
sf = StripForces('forces_rotor.h5', r_tip=0.125, rpm=6000)
result = sf.compute(span_min=0.02, n_span_bins=20)

phase_locked = sf.phase_lock(result, dt=0.000056, n_azimuth_bins=72)
sf.plot_vs_angle(phase_locked, component='axial', strips=[0, 4, 9, 14, 19],
                  savepath='strip_vs_angle.png')  # polar by default; polar=False for Cartesian
```

Needs roughly a full revolution of frames at minimum to have every
azimuth bin populated; several revolutions is what makes the averaging
part actually do anything (fewer just assigns each frame its own bin
with nothing to fold together).

Azimuth assumes constant rpm across the file (`azimuth = (t * rpm * 6) mod 360`.

### Harmonics (Hanson's method's actual input): `harmonics()` / `plot_harmonics()`

FFT of each strip's force time history, reported at harmonics of the
rotor's own rotation frequency (1P, 2P, 3P, ...). They are `|F_n(r)|`,
the loading-harmonic input Hanson's tonal noise method needs.

```python
h = sf.harmonics(result, dt=0.000056, component='axial', n_harmonics=17)
sf.plot_harmonics(h, strips=[0, 4, 9, 14, 19], savepath='strip_harmonics.png')
```

Uses a plain FFT over the whole time series (not Welch's method like
`SurfaceVariable.periodogram()`). The file should span close to an
integer number of full revolutions.

#### Phase: `harmonics(return_phase=True)` / `peak_azimuth()` / `reconstruct_from_harmonics()`

Off by default (most exploratory work), `harmonics(..., return_phase=True)`
also returns each harmonic's phase [rad], needed for two things:

1. **Hanson's model itself:** a magnitude-only harmonic can't be summed
   back into a physically meaningful signal, and the interference
   between radial stations' contributions (which shapes the final
   directivity) depends on their relative phase, not just magnitude.
2. **Tying a harmonic back to a physical cause, independent of the noise
   calculation:** `peak_azimuth()` converts phase into the azimuth
   [deg] where each harmonic's own contribution peaks
   (`-phase/n mod (360/n)`, since a harmonic `n` repeats `n` times per
   revolution).

`reconstruct_from_harmonics()` rebuilds an azimuth-domain curve from
magnitude + phase (`sum_n magnitude_n * cos(n*phi - phase_n)`, DC
excluded since `harmonics()` detrends it away) an overlay this against
`phase_lock()`'s own empirical folded curve as a validation check: if a
handful of harmonics already reconstructs the real curve closely, that
confirms the FFT decomposition captured the dominant unsteady content
(and tells you honestly how many harmonics actually matter for this
case), rather than trusting magnitudes/phases blind.

```python
h = sf.harmonics(result, dt=0.000056, component='axial', n_harmonics=17, return_phase=True)
sf.plot_harmonics(h, strips=[0, 4, 9], show_phase=True, savepath='strip_harmonics_phase.png')

peak_deg = sf.peak_azimuth(h)  # (n_harmonics, n_span_bins)

phase_locked = sf.phase_lock(result, dt=0.000056, n_azimuth_bins=72)
az, recon = sf.reconstruct_from_harmonics(h, azimuth_deg=phase_locked['azimuth_deg'])
# overlay recon[:, i] + result['axial'][:, i].mean() against phase_locked['axial'][:, i]
```

**Downstream output**: `save_harmonics(h, filepath)` writes `radius`,
`chord` (if chord-subdivided), `harmonic`, `magnitude`, and `phase` (if
present) to one self-contained `.h5` file giving the actual Hanson-model-ready
artifact, usable without this class or the original `.snc`-derived file
again.

## What's still open

- Iso-radius / (r/R, x/c) resampling directly from raw `.snc` surfel
  clouds (needed for skin-friction-line or Cp-vs-chord plots) isn't solved
  yet - see `bladeprocessor/` for the grid-based tools that work on
  already-resampled `SurfaceField`-style HDF5 files instead.
- The additive constant for converting raw lattice `Static Pressure` to Pa
  remains unresolved and is not assumed to be case-independent - hence the
  hard requirement to use `pf2ens` for pressure rather than the raw file.
- `FrictionLines`' span/chord axes are raw Cartesian columns
  (`span_axis`/`chord_axis`), not derived from the rotor's rotation axis,
  a rotation-axis-based derivation was tried and abandoned (see the class
  docstring) because it assumes the chord line lies in the rotor disk
  plane, which breaks on any blade with real geometric pitch/twist. The
  Cartesian fallback works for every case in this project so far but
  isn't automatically correct for a differently-oriented mesh, and
  doesn't by itself handle a twisted blade's true local chord direction
  either - a proper fix (e.g. per-station PCA of the point cloud) is not
  implemented.
- `SurfaceVariable`'s `reverse_chord` (which end of the raw chord axis is
  the leading vs. trailing edge) has no automatic detection, it's
  arbitrary per case, currently determined by eye (checking whether the
  Kutta-condition/near-zero-Cp end and the stagnation-point/suction-peak
  end land where expected) rather than computed from anything in the
  file. `SurfaceVariable.cp(stat='rms')` is implemented and its reduction
  formula is validated synthetically, but hasn't yet been checked against
  a real multi-frame pressure file.
