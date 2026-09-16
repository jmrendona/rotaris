# Handoff: rotaris post-processing toolkit — session context

This file summarizes context for a fresh Claude Code session (e.g. on the
HPC, where the real multi-frame `.snc`/pressure data lives) to pick up
without starting cold. `README.md` is the canonical reference for *how to
use* everything below — this file is about *why things are the way they
are*, what's been validated against real vs. synthetic data, and what's
still open. Kept lean on purpose: resolved investigations are compressed
to a short summary once confirmed working in production, not kept as full
blow-by-blow evidence trails.

## OPEN INVESTIGATION — `stagnation_line()` clusters near the TE, not the LE, on its first real run

**Status**: `stagnation_line()` (`bladeprocessor/surface_variable.py`,
`SurfaceVariable`) was previously only validated on a synthetic case (see
below). Its first run against a real instantaneous pressure file produces
a clean, plottable result, but it clusters near the TRAILING edge instead
of the LEADING edge, where the physical stagnation point actually has to
sit. Suspected physical cause (user's own hypothesis): a real Kutta-
condition-related pressure feature near the TE getting mistaken for the
LE stagnation peak. **More likely cause given this project's own
history**: an orientation bug, not a physical one - see below. **Needs
the real HPC file to resolve** - nothing here can be confirmed or fixed
blind on the local machine.

### Primary hypothesis: `reverse_chord` orientation

`stagnation_line()` restricts its peak search to `x/c <= search_xc_max`
(default 0.2), where `x/c` is this file's own ARBITRARY local chord
normalization (`(chord - c_min)/(c_max - c_min)`, flipped by
`reverse_chord`) - there is no independent geometric check that `x/c=0`
is actually the true leading edge. This is the EXACT SAME ambiguity
already found and fixed for `Cf` on this project's own case:
`reverse_chord=True` was required (see "The x/c orientation..." below) to
match a trusted PowerVIZ reference so the Cf peak lands at `x/c=0`. If
`stagnation_line()` was called with a different (or default `False`)
`reverse_chord` than what's already established for this case, its
"near x/c=0" window is actually sampling near the GEOMETRIC TRAILING
edge - which reproduces exactly this symptom.

**Check this first, no new data/code needed**: was `reverse_chord` set
the same way it already had to be for `cf_at_radii()`/`plot_cf_radii()`/
`friction_lines()` on this case? If not, flip it and rerun - this alone
may be the entire fix.

### Secondary hypothesis: blade twist reversing LE/TE at some span stations even with the right global flag

`reverse_chord` is one GLOBAL flag - it assumes one raw chord extreme is
consistently the LE at EVERY span station. Enough geometric twist can
break that locally (root/tip typically worst). If the primary fix works
at MOST stations but specific radii are still TE-side, this is why - a
single global flag can't fix a station-by-station flip.

### Diagnostic to run on the HPC (needed either way)

```python
from bladeprocessor.surface_variable import SurfaceVariable

sv = SurfaceVariable(inst_pressure_file, r_tip=0.125, rho_ref=1.22523, rpm=6000, pref=101325)

# search_xc_max=1.0 - NO near-LE restriction, so this reports wherever
# Cp's actual peak sits, letting the data reveal the true orientation:
for reverse in (False, True):
    pts = sv.stagnation_line(stat='mean', span_min=0.02, search_xc_max=1.0, reverse_chord=reverse)
    print(f"--- reverse_chord={reverse} ---")
    for p in pts:
        print(f"span={p['span']:.4f}  surface={p['surface']}  xc={p['xc']:.3f}  cp={p['cp']:.3f}")
```

Report back the pattern of `xc` across span, for both settings:

- If ONE setting gives `xc` consistently near 0 (not 1) at EVERY span
  station → primary hypothesis confirmed - that's the right
  `reverse_chord` for this case; use it with the default
  `search_xc_max=0.2` from here on.
- If `xc` is near 0 at SOME stations and near 1 at OTHERS for BOTH
  settings (no global flag makes it uniform) → secondary hypothesis
  confirmed (twist-driven per-station flip) - report which span ranges
  land on which side. Fix would need per-bin LE/TE detection from the
  rotor's own known rotation direction (local relative-velocity
  direction via `lrf_axis_direction`/angular velocity sign, dotted
  against each bin's actual 3D chord vector) instead of one global flag
  - not implemented, needs this case's real geometry/rotation to design
  and verify against.
- Also confirm `span_min=0.02` (or this case's own blade-isolation value)
  was actually used - the two-blade-mixing bug (below) gives a
  DIFFERENT, distinguishable symptom (two spurious peaks, one near each
  chord extreme) but is worth ruling out on a first real run of any
  x/c-based method.

## RESOLVED (compressed - full evidence trails available in git history if ever needed again)

- **`Surface_X/Y/Z-Force` global-vs-LRF reference frame bug**: PowerFLOW
  stores this in the GLOBAL (lab) frame while `Geometry/*` (and
  everything chordwise/spanwise/radial/tangential built from it) is in
  the LOCAL (LRF, blade-co-rotating) frame. Fixed in `SNCReader.to_h5()`/
  `_write_surfel_group()` (`converters/snc_reader.py`) - rotates
  `Surface_X/Y/Z-Force` about `lrf_axis_direction` by `-angle(frame)`
  before writing, `angle(frame) = lrf_initial_angular_rotation +
  lrf_constant_angular_vel_mag * start_time[frame]` (`start_time` is an
  ABSOLUTE simulation clock, not reset per file - an earlier version
  wrongly subtracted `start_time[0]`, caught via the user's own question
  about files starting at different offsets). Prefers
  `parse_nc_stats()`'s PowerFLOW-authoritative `lrf_position_rad` (via
  `_aligned_frame_meta()`, matched by absolute `start_time`, not row
  index) when `nc_stats_path` is given. `to_h5(..., blade_lrf_offset_deg=0.0)`
  exists for an independently-confirmed fixed mounting misalignment (0 by
  default). In production use since - all later work (torque, harmonics,
  convergence checks) relies on this being correct.
- **`_LargeRecordNetcdfFile` 32-bit ceilings** (`converters/snc_reader.py`):
  two separate scipy/NumPy 32-bit limits (the NetCDF `vsize` header
  field, and a NumPy structured-dtype per-field ceiling) hit on a
  fine-DNS-mesh multi-GB `.snc` file. Fixed by recomputing the true
  per-record size independently of the file's own `vsize`, and never
  building a structured dtype for record variables (each gets its own
  plain strided view). Validated byte-for-byte identical to plain scipy
  on real files up to 186 GB. **`mmap=True`** (not the original `False`)
  is also now used - `mmap=False` read the ENTIRE record block into RAM
  at file-open time regardless of what's actually used, which OOM-killed
  a real ~660 GB conversion job; `mmap=True` uses the identical
  strided-view logic, backed by lazily-paged memory instead.
- **`FrictionLines`/`StripForces` OOM on a whole-rotor (no separate blade
  parts) case**: both eagerly loaded the ENTIRE per-frame force field
  into RAM at construction. Fixed by adding `span_min`/`span_max` at the
  CONSTRUCTOR level (not just per-call) - crops the point selection
  BEFORE reading `Data/*` off disk (masked HDF5 column reads). Centering
  for `_span_chord()` is computed from the FULL, uncropped extent first
  (stored, not recomputed from the cropped survivors) so existing
  per-call `span_min`/`span_max` downstream stays consistent.
  `span_min=0.02` (isolating one blade half - already required everywhere
  downstream on this whole-rotor case) is the value used in `manager.py`.
- **Cf-magnitude "growth" investigation**: apparent smooth growth with
  frame index was genuine once-per-revolution PERIODICITY (frame 720 ≈
  frame 0, four revolutions later), not a bug or an unbounded transient -
  confirmed across all 829 frames of a real case.

## The x/c orientation & two-blade-mixing bug pattern (recurring theme — directly relevant to the OPEN INVESTIGATION above)

1. **Two-blade mixing**: this project's `.snc` files lump the WHOLE rotor
   (both blades) into one face (`/Rotor::Default-Segment`) with no
   per-blade tag. Any method selecting "points near radius r" without
   also cropping by `span_min`/`span_max` silently mixes both blades'
   surfels, corrupting local x/c normalization (produces a spurious extra
   peak at both x/c=0 AND x/c=1 instead of one real peak). **Always pass
   `span_min`/`span_max` on this project's own data** - `span_min=0.02`
   isolates one blade half (span runs symmetrically ~-0.125 to +0.125 m,
   hub cluster within roughly ±0.02 m).
2. **LE/TE orientation ambiguity**: x/c=0 is arbitrarily assigned to
   whichever raw chord extreme happens to be the minimum value - no
   inherent physical meaning. `reverse_chord=True` was needed on this
   project's own case to match a trusted PowerVIZ reference where the Cf
   peak sits at x/c=0. Same ambiguity exists for Cp/`stagnation_line()`
   (`SurfaceVariable`) - and is the leading suspect for the OPEN
   INVESTIGATION above.

## Known open risk: blade sweep (colleague's automotive cooling fan case)

Not yet tested against any swept-blade data.

- **Robust to sweep, no fix needed**: `SNCReader` conversion,
  `surface_split()` (normal-based, purely local), radius computation
  (`_radius()`, true distance from rotation axis), `StripForces`'s entire
  pipeline (built on the rotation axis, never chord direction).
- **At risk under sweep**: anything using a THIN RADIUS BAND as a proxy
  for "one aerodynamic station's full chord, LE to TE" -
  `cf_at_radii()`/`plot_cf_radii()`, `separation_line()`,
  `migration_line()`, `StripForces`'s `n_chord_bins`. Sweep means a
  station's LE and TE genuinely sit at different true radii, so a
  constant-radius band either misses them (`tol` small) or mixes in
  neighboring stations (`tol` widened) - same symptom as two-blade
  mixing, different cause.
- **Proposed fix, not yet built** (needs real swept data to develop
  against): replace the constant-radius band with one that follows the
  blade's actual swept reference line - either an analytic shear
  correction if the sweep angle is known, or numerically estimated from
  the mesh itself (track chord-centroid shift per span slice, fit a
  reference line, band by distance from that curve). Get one real file
  from the colleague first, check how much this actually matters, before
  building either blind.

## Who's who / working style

- User is a PhD student running PowerFLOW rotor simulations, building
  this toolkit for real analysis + eventually feeding a colleague's
  swept-blade automotive-cooling-fan case, and for tonal-noise prediction
  inputs (Hanson's method).
- Strong preference: validate everything against real data before
  trusting it; when real data isn't available, synthetic validation is
  fine BUT must be explicitly and unambiguously flagged as synthetic -
  don't let a synthetic demo look like it could be a real result.
- Whenever something is generated/computed, copy the resulting
  image/file to the user's own visible folder (not just the sandbox
  scratchpad): `/Users/jmrendona/OneDrive - USherbrooke/PhD/rotor-alone/6e-5-6000rpm/images/test/`
  (note: different OneDrive root path, WITH spaces around the dash, than
  this repo's own `OneDrive-USherbrooke` working directory).
- Plot style: LaTeX/Computer Modern fonts, `cividis` colormap for
  multi-curve plots, `axes.labelsize=18`, `legend.fontsize=18`, no
  decorative titles, single-line plots in black (not the matplotlib
  default blue), `dpi=600` default on every `savefig`, simple
  math-notation axis labels with units.
- OneDrive on the local machine repeatedly evicts files to "dataless"
  placeholders mid-session (0 bytes, read raises `TimeoutError`/hangs) -
  fix is always a retry loop around `open(path,'rb').read()` with a few
  seconds' sleep between attempts. Not a real error - the HPC filesystem
  doesn't have this problem.

## Module map (`bladeprocessor/`, `converters/`)

- `converters/snc_reader.py` (`SNCReader`) - reads a raw PowerFLOW `.snc`
  surface file (NetCDF), converts `Surface_X/Y/Z-Force`/`Skin_Friction`
  to physical units (Pa). `Static_Pressure` is NOT converted here (needs
  `pf2ens`'s internal Cp-based translation - see `converters/ensight_to_h5.py`).
  `to_h5(..., surface_split=True)` splits Upper/Lower by surfel normal
  sign relative to the rotation axis.
- `bladeprocessor/friction_lines.py` (`FrictionLines`) - wall shear/Cf
  from the forces branch. Span/chord/thickness axes are RAW CARTESIAN
  columns (`span_axis`/`chord_axis`/`thickness_axis`, default 0/2/1), NOT
  derived from the rotation axis (a rotation-axis-derived chord broke
  under real blade pitch/twist). Radius IS derived from the true
  rotation axis and is robust regardless of blade shape.
- `bladeprocessor/surface_variable.py` (`SurfaceVariable`) - generalizes
  "any variable at radii" beyond Cf to Cp, y+, RMS statistics, pressure
  fluctuation, point time traces + Welch periodograms, and
  `stagnation_line()` (see OPEN INVESTIGATION). Works against either the
  forces branch or the `pf2ens`-derived pressure branch.
- `bladeprocessor/surface_field.py` (`SurfaceField`,
  `SurfaceFieldComparator`) - works on already-resampled `(Radius,
  Chord)` grid files. Known pre-existing edge case: `plot_delta()` fails
  if a delta is exactly zero everywhere (contour levels non-increasing) -
  not fixed, only hit in a degenerate self-comparison test.
- `bladeprocessor/strip_forces.py` (`StripForces`) - per-radial-strip
  time-resolved axial/radial/tangential force (Hanson's method input).
  Physical basis built purely from the rotation axis - `axial`/`radial`/
  `tangential`, two sign ambiguities (`flip_axial`/`flip_tangential`,
  same "which raw direction is which" issue as `reverse_chord`).
- `bladeprocessor/convergence.py` - generic (not tied to any one class)
  running-statistics convergence-checking tools; see README.md for the
  full API. Consumes a plain 1D per-frame scalar series from any class -
  `StripForces.total_loads()`, `FrictionLines.cf_time_series()`, or
  `SurfaceVariable.variable_time_series()`/`cp_time_series()`.

## Pending / next steps (in likely priority order)

1. Resolve `stagnation_line()`'s TE-clustering (see OPEN INVESTIGATION
   above) - needs the real HPC pressure file.
2. Get a real file from the colleague's swept-blade case, check the
   sweep risk empirically (see above) before trusting any
   `cf_at_radii()`-family result on it.
3. Possible future methods discussed but NOT built (only if the user
   wants them): proper streamline integration (`matplotlib.streamplot`)
   for skin-friction topology; cross-validating `separation_line()`
   against `critical_points()` (separation lines are topologically
   required to emanate from saddle points); sectional `c_l`/`c_d` from
   strip forces + local chord/`q_ref`; blade geometry (chord/thickness/
   twist) extraction for Hanson's model's non-loading inputs; flap/lag
   bending moments about a hinge.

## File/data conventions specific to this project's own validation case

- `r_tip=0.125` m, `rho_ref=1.22523` kg/m³, `rpm=6000` for the isolated-
  rotor-in-hover case.
- `span_min=0.02` isolates one blade half (whole-rotor `.snc`, no
  per-blade face tag); `reverse_chord=True` needed to match the trusted
  Cf/Cp LE-at-x/c=0 convention - **verify this is still right for
  `stagnation_line()` too, see OPEN INVESTIGATION above** (it was only
  ever confirmed via a Cf reference plot, never independently for Cp).
