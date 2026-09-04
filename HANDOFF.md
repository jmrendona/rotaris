# Handoff: rotaris post-processing toolkit — session context

This file summarizes an extended collaborative session building out the
`rotaris` toolkit, so a fresh Claude Code session (e.g. on the HPC, where
the real multi-frame `.snc` data lives) can pick up with full context
instead of starting cold. `README.md` is the canonical reference for
*how to use* everything below — this file is about *why things are the
way they are*, what's been validated against real vs. synthetic data,
and what's still open.

## OPEN INVESTIGATION — read this first, work is moving to the HPC mid-debug

**Symptom**: on a real, transient (~400-frame), already-past-transient
(user confirmed thrust converged by ~5 revolutions) `.snc` case
(`6e-5-6000rpm`), `FrictionLines.plot_cf_radii()`'s Cf magnitude grows
smoothly and monotonically with frame index, then saturates: peak Cf
~0.044 at frame 0, ~0.23 at frame 10, ~0.83 at frame 40, ~1.1 at frame
80 (images in `.../images/cf_test/cf_radii_frame{0,10,40,80}.png`, and
the two originally-reported `cf_radii_avg.png` vs `cf_radii_frame0.png`
that kicked this off, another ~13x gap in the same direction). The
radial trend also inverts between early and late/averaged frames (which
radius has the biggest LE peak flips). Separately: averaging the raw
forces over frames [0,1] vs [50,51] gives nearly IDENTICAL thrust
(1.319 N vs 1.313 N) but torque that changes a lot and flips sign
(+0.0219 N·m vs -0.0114 N·m).

**Why thrust-stable-but-Cf/torque-not-stable is NOT inherently
contradictory** (resolved, not open): `tau = F - (F.n)n` mathematically
removes the ENTIRE pressure contribution to `F` (pressure acts purely
along the local normal `n` by definition, so it contributes exactly
zero to `tau`) - only the viscous/shear part of `F` survives into Cf.
Thrust is essentially the rotation-axis component of `F`, dominated by
the (fast-converging) pressure part. Torque includes both a pressure
part AND a viscous/profile-drag part, so it's the natural first place a
still-transient viscous field would show up even while thrust looks
converged. This is real physics, not a bug - but it only explains why
thrust and Cf/torque CAN move independently, it does NOT by itself prove
the observed Cf growth is real physics rather than a reader bug. Both
explanations below are still live.

**Two live hypotheses, not yet distinguished**:

1. **Real, still-unconverged near-wall/viscous statistics.** Skin
   friction (SGS-turbulence-driven) is well known to converge much more
   slowly than integrated pressure loads in scale-resolving CFD - the
   user's "5 revolutions, transient removed" check very likely confirmed
   THRUST had converged, not necessarily this finer near-wall quantity.
   The observed shape (smooth, monotonic, SATURATING - only 1.32x growth
   from frame 40→80 despite doubling the frame count - and each
   individual frame's spatial Cf-vs-x/c shape is smooth/physically
   sane, an LE peak decaying to the TE, not scrambled noise) is
   consistent with this: a byte-misalignment/stride bug would be
   expected to scramble the SPATIAL structure within a frame (reinterpret
   floats essentially at random), not preserve a clean airfoil-shaped
   curve while just scaling its amplitude up smoothly.
2. **A real bug in `_LargeRecordNetcdfFile`** (see next section) on
   THIS file's specific layout. Motivating concern: this reader was
   rewritten from scratch this session and is used UNCONDITIONALLY for
   every `.snc` file now (no size-based branch), but was only validated
   against hand-built SYNTHETIC files (single record var; two record
   vars, one big one small; a byte-padding case) - never against this
   actual file. If this file's real record-variable layout has some
   property none of those synthetic tests happened to cover, a
   systematic (not random) stride/offset error could plausibly still
   preserve smooth-looking per-frame data while being wrong in a way
   that correlates with frame index.

**Decisive test, not yet run**: `diagnose_snc.py` (repo root - written
this session, needs to travel to the HPC with everything else). Reads
the raw `.snc` directly via `SNCReader` (our patched reader) AND, if
this file is small enough for it to work at all, via plain unmodified
`scipy.io.netcdf_file` side by side, for a handful of frames
(0/1/10/40/80/150). Run as:

```bash
python diagnose_snc.py /path/to/6e-5-6000rpm/case.snc
```

- If plain scipy can open the file (likely - this case predates the
  8 GB DNS case that originally motivated `_LargeRecordNetcdfFile`, so
  it may not even need the fix) AND its numbers match `SNCReader`'s
  frame-by-frame: our reader is exonerated, this is hypothesis 1 (real
  convergence question) - stop suspecting the code, go look at the
  solver's own convergence history for near-wall quantities specifically.
- If they DON'T match: hypothesis 2 is confirmed - fix
  `_LargeRecordNetcdfFile`, most likely the assumption (in its
  docstring) that the first-declared record variable in the header also
  has the smallest on-disk offset (a NetCDF-classic-format invariant
  that should hold, but hasn't been independently confirmed for
  whatever wrote THIS file), or something about the per-variable stride
  math that this file's specific record-variable layout exercises
  differently from the three synthetic cases already tested.
- Also worth doing regardless of outcome: if an OLDER, pre-this-session
  conversion of this exact case (or a similar one) still exists from
  before `_LargeRecordNetcdfFile` existed, compare its Cf plots against
  the new ones for the same frames - a clean before/after regression
  check that doesn't require any new HPC runs.

## `SNCReader` / `_LargeRecordNetcdfFile` (`converters/snc_reader.py`) — two 32-bit ceilings, fixed this session, UNVALIDATED against most real files

Built to fix `ValueError: read length must be non-negative or -1` on an
8 GB, fine-DNS-mesh, single-frame `.snc` file that plain
`scipy.io.netcdf_file` couldn't open. Two SEPARATE bugs, both in scipy
itself, both fixed by a custom `_LargeRecordNetcdfFile(sio.netcdf_file)`
subclass that overrides only `_read_var_array()`:

1. **NetCDF's own 32-bit `vsize` header field** overflows/gets
   corrupted (scipy's SIGNED 32-bit parse misreads the format's own
   documented `2^32-1` escape sentinel as `-1`) once a record variable's
   true per-record byte size crosses ~2.1-4.3 GB. Fixed by recomputing
   the true per-record size independently from each variable's own
   (unaffected) shape/dtype instead of trusting the file's stored
   `vsize`.
2. **A SEPARATE NumPy ceiling**, surfaced only after fixing #1: scipy
   reads every record variable through ONE NumPy STRUCTURED dtype (one
   "field" per record variable), and NumPy's structured-dtype machinery
   caps a single field's byte size at a C `int` (~2.1 GB) - hit on the
   same 8 GB file even after #1 was fixed, because "measurements" is a
   multi-GB single field regardless. Fixed by never building a
   structured dtype for record variables at all: read the whole
   interleaved record block once as raw bytes, then hand each record
   variable its OWN plain strided view into it (using that variable's
   own file offset - `begin_`, already correctly parsed - as the anchor,
   and the corrected per-record size from #1 as the between-records
   stride). An EARLIER, narrower version of this fix only handled the
   case of exactly one record variable, which turned out not to match
   this file's real layout (it has more than one - "measurements" plus
   at least one smaller one riding the same frame axis, e.g. a
   timestamp) and hit the exact same ceiling through the leftover
   fallback path - the current version has no such precondition.

`SNCReader.__init__` uses `_LargeRecordNetcdfFile` UNCONDITIONALLY, for
every file regardless of size - there is no runtime check/branch between
"the two readers"; it's a strict replacement, validated to behave
identically to plain scipy on ordinary files (see below), so there's no
downside to always using it.

**Validated**: both fixes reproduced against the EXACT reported error
messages (constructing the same shape of structured dtype scipy would
have built; surgically corrupting a real small NetCDF file's `vsize`
field to the documented sentinel value). Correctness validated with
THREE hand-built synthetic multi-record-variable files (a single record
var; two record vars of very different size sharing the frame axis,
mirroring this file's apparent layout; a byte-alignment padding case
with an odd-sized int16 record var) - `_LargeRecordNetcdfFile` matched
plain scipy byte-for-byte in every case, no regression.

**NOT yet validated end-to-end against any real multi-GB `.snc` file**
(the DNS case that motivated this needs the HPC to even test) - and,
per the OPEN INVESTIGATION above, its correctness against THIS project's
own everyday-sized `6e-5-6000rpm` case is now an active, unresolved
question, not a settled assumption. Treat any Cf/friction-line result
from a freshly-(re)converted `.snc` file as suspect until
`diagnose_snc.py` clears this reader on that file.

## `SurfaceVariable.stagnation_line()` — leading-edge stagnation point tracking (new this session)

Motivated by potential-flow interaction with a downstream obstruction
(strut/stator vane): a local AoA change shifts the stagnation point off
the geometric LE, toward whichever surface sees the higher effective
incidence. `stagnation_line()` sweeps span in bins (same architecture as
`FrictionLines.separation_line()`/`migration_line()`), and in each bin
searches the local Cp MAXIMUM near `x/c=0` across BOTH surfaces combined
(unlike separation/migration lines, which each search one surface
already split by wall-normal sign - the whole point here is seeing
WHICH side it lands on). Returns a signed `x/c` (`+` on Lower, `-` on
Upper) per span bin - one continuous scalar, directly plottable via the
new `plot_stagnation_line()` (label→points dict, e.g. comparing several
individual frames against the mean - the "does it move frame to frame"
question). `plot_variable_surface(show_stagnation_line=True)` overlays
it directly on the blade-contour subplots instead, jumping between the
Upper/Lower panels as it migrates sides - only valid with both surfaces
plotted (raises clearly otherwise). `save_stagnation_line()` matches the
existing `save_separation_line()`/`save_migration_line()` text-export
convention.

**Validated** against a synthetic case with an engineered Cp peak at
known (surface, x/c) locations at several span stations (including one
placing the true peak exactly on the geometric LE) - recovered the
correct surface and x/c to four decimal places at every station.
`plot_variable_surface(show_stagnation_line=True)` and the single-surface
guard both confirmed to run without error. **Not yet run against a real
downstream-obstruction case** (needs one where this interaction is
actually present - none available yet).

## Who's who / working style

- User is a PhD student running PowerFLOW rotor simulations, building
  this toolkit for real analysis + eventually feeding a colleague's
  swept-blade automotive-cooling-fan case, and for tonal-noise
  prediction inputs (Hanson's method).
- Strong preference: validate everything against real data before
  trusting it; when real data isn't available, synthetic validation is
  fine BUT must be explicitly and unambiguously flagged as synthetic
  (this came up directly - don't let a synthetic demo look like it
  could be a real result; say so clearly, every time).
- Whenever something is generated/computed, copy the resulting
  image/file to the user's own visible folder (not just the sandbox
  scratchpad) so they can actually look at it:
  `/Users/jmrendona/OneDrive - USherbrooke/PhD/rotor-alone/6e-5-6000rpm/images/test/`
- Plot style conventions established over the session: LaTeX/Computer
  Modern fonts, `cividis` colormap (not viridis/tab10) for anything with
  multiple radius-colored or strip-colored curves, `axes.labelsize=18`,
  `legend.fontsize=18`, no decorative titles (`fig.suptitle`/generic
  `ax.set_title` calls were deliberately removed), simple math-notation
  axis labels with units (`$C_f$ [-]`, `$x/c$ [-]`, `Force [N]`, etc).
- OneDrive on this machine repeatedly evicts local files to "dataless"
  placeholders mid-session (shows as 0 bytes, read raises `TimeoutError`
  or hangs) - the fix is always the same: retry a raw
  `open(path,'rb').read()` in a loop with a few seconds' sleep between
  attempts until it materializes. Not a real error, just routine friction
  on this machine - the HPC filesystem presumably won't have this problem.

## Module map (`bladeprocessor/`, `converters/`)

- `converters/snc_reader.py` (`SNCReader`) - reads a raw PowerFLOW `.snc`
  surface file (NetCDF), converts `Surface_X/Y/Z-Force` and
  `Skin_Friction` to physical units (Pa) using the file's own lattice
  scale factors (validated ~0.999 correlation against the file's own
  stored Skin Friction). `Static_Pressure` is NOT converted here (needs
  `pf2ens`'s internal Cp-based translation - see `converters/ensight_to_h5.py`).
  `to_h5(..., surface_split=True)` splits Upper/Lower by surfel normal
  sign relative to the rotation axis - robust, not affected by blade
  sweep or twist (it's a purely local geometric quantity from the mesh
  itself). Geometry dataset names are `Normal_X`/`Normal_Y`/`Normal_Z`
  (renamed from `NX` this session - any `.h5` converted with the OLD
  `SNCReader` code still has the old `NX` key and will need
  reconverting to work with the current `FrictionLines`).
- `bladeprocessor/friction_lines.py` (`FrictionLines`) - wall
  shear/Cf from the FORCES branch (`SNCReader.to_h5()`, not
  `pf2ens`/pressure). Span/chord/thickness axes are RAW CARTESIAN
  columns (`span_axis`/`chord_axis`/`thickness_axis`, default 0/2/1),
  NOT derived from the rotation axis - a rotation-axis-derived chord was
  tried and abandoned early on because it broke under real blade
  pitch/twist. Radius, however, IS derived from the true rotation axis
  (`_radius()`) and is robust regardless of blade shape.
- `bladeprocessor/surface_variable.py` (`SurfaceVariable`) - generalizes
  the "any variable at radii" pattern beyond Cf to Cp, y+, RMS
  statistics, pressure fluctuation, point time traces + Welch
  periodograms. Works against either the forces branch or the
  `pf2ens`-derived pressure branch (same schema).
- `bladeprocessor/surface_field.py` (`SurfaceField`,
  `SurfaceFieldComparator`) - works on already-resampled `(Radius,
  Chord)` grid files, not raw `.snc` surfel clouds. Has a known
  pre-existing edge-case bug: `plot_delta()` fails if a delta is exactly
  zero everywhere (contour levels become non-increasing) - not fixed,
  out of scope, only hit in a degenerate self-comparison test.
- `bladeprocessor/strip_forces.py` (`StripForces`) - built this session,
  see below, the largest single piece of new work.

## The x/c orientation & two-blade-mixing bug pattern (recurring theme)

Found and fixed multiple times this session, same root causes each time:

1. **Two-blade mixing**: this project's `.snc` files lump the WHOLE
   rotor (both blades) into one face (`/Rotor::Default-Segment`) with no
   per-blade tag. Any method that selects "points near radius r" without
   also cropping by `span_min`/`span_max` silently mixes both blades'
   surfels together, corrupting local x/c normalization (confirmed:
   produces a spurious extra peak at both x/c=0 AND x/c=1 instead of one
   real peak). Fixed in `cf_at_radii()`, `plot_cf_radii()`,
   `separation_line()`, `migration_line()`; `friction_lines()` already
   had it. **Always pass `span_min`/`span_max` on this project's own
   data** - `span_min=0.02` isolates one blade half on the case used for
   validation this session (span runs symmetrically ~-0.125 to +0.125 m,
   hub cluster sits within roughly ±0.02 m).
2. **LE/TE orientation ambiguity**: x/c=0 is arbitrarily assigned to
   whichever raw chord extreme happens to be the minimum value - no
   inherent physical meaning. `reverse_chord=True` was needed on this
   project's own case to match a trusted PowerVIZ reference
   (`images/cf/skin-friction_radii_plot.png`) where the Cf peak sits at
   x/c=0. Same ambiguity exists for Cp (`SurfaceVariable`).

## `separation_line()` / `migration_line()` / `critical_points()`

Skin-friction-line topology tools on `FrictionLines`.

- `separation_line()`: chordwise-Cf zero crossings. Labeled by ORDER
  along x/c (1st/3rd/5th = `'separation'`, 2nd/4th/6th =
  `'reattachment'`), NOT by raw sign - chordwise Cf's sign has no fixed
  physical meaning (depends on arbitrary `chord_axis` orientation).
- `migration_line()`: spanwise-Cf zero crossings (radial near-wall
  migration direction reversal - centrifugal pumping vs. inward
  migration - a DIFFERENT physical phenomenon from separation). Labeled
  `'outward'`/`'inward'` by actual sign, since span (unlike chord) is a
  raw absolute coordinate with real physical meaning once
  `span_min`/`span_max` picks one blade half.
  **Real debugging story worth knowing**: an early unfiltered version
  showed dense LE noise (near-zero-magnitude, rapidly alternating sign,
  not real structure). Two amplitude-based filters to suppress it both
  backfired (per-bin local-max scale asymmetrically killed one side of
  genuine pairs → all-one-kind output; global scale still got dominated
  by the LE's own large gradient → collapsed to one LE-adjacent crossing
  per bin). The fix that worked: exclude by POSITION (`edge_crop`, same
  idea as `SurfaceVariable.at_radii()`'s parameter), not by amplitude.
- `critical_points()`: locations where the WHOLE wall-shear vector
  vanishes (not just one component) - classified via local Jacobian
  eigenvalues into `'node'`/`'saddle'`/`'focus'` (Tobak & Peake 1982
  topology). `'focus'` = genuine vortex core footprint (complex
  eigenvalues = rotation in the linearized flow near the point) - the
  ONLY one of the three with that signature. `poincare_index()` sums
  indices (`N+F-S`, node/focus=+1, saddle=-1) - only required to equal 2
  (Poincaré-Hopf) on a CLOSED surface; this project's own selections are
  open/cropped patches, so it's a diagnostic number, not a pass/fail
  check.

All three validated on the real converted rotor case (`span_min=0.02`,
`reverse_chord=True`) and found REAL, physically coherent structure
clustered in the same tip region (span ~0.095-0.125 m) - three
independent methods agreeing is the main confidence-builder here.

## `StripForces` (`bladeprocessor/strip_forces.py`) — the big new piece

Built this session to replace a manual, per-frame PowerVIZ "Force
Graph" CSV export (`converters/forces_strip.py`'s `ForcesCSVConverter`,
which STATISTICALLY GUESSES which exported color channel is
axial/tangential/radial via a mean/std heuristic - fragile).
`StripForces` derives axial/radial/tangential directly from the known
rotation axis geometry instead - no guessing.

- **Physical basis**: `axial` = rotation axis direction (constant),
  `radial` = unit vector from axis to surfel (unambiguous), `tangential`
  = `axial x radial`. Two genuine sign ambiguities (`flip_axial`,
  `flip_tangential`) - same "which raw direction is which" issue as
  `reverse_chord` elsewhere, since `lrf_axis_direction`'s sign is
  whatever the file happens to have.
- `compute()` - per-strip (optionally per-strip-per-chord-bin via
  `n_chord_bins`, for non-compact-chord cases) time-resolved
  axial/radial/tangential force, EVERY frame. Embeds `result['totals']`
  (see `total_loads()`) computed from the exact same surfel selection,
  guaranteed consistent with whatever `span_min`/`span_max` was used.
- `total_loads()` / `result['totals']` - thrust (`Sum(F_axial)`), torque
  (`Sum(F_tangential_i * radius_i)` - a MOMENT, needs the per-surfel
  radius lever arm, NOT the same as summing tangential force),
  radial_force, tangential_force. **Validated against the user's own
  independently known total** (2.6 N thrust, full 2-bladed rotor):
  computed 1.319 N for one blade - matches half almost exactly. Torque
  similarly confirmed ~half the user's known total.
- `plot_bar_forces()` - bar + cumulative curve, styled to match the
  user's own PowerVIZ reference (`images/forces/bar_forces/Force-Graph-1-bar_forces.png`).
  `normalize_radius` (default True, r/R) optional. `show_totals=True`
  annotates the figure - dimensional values normally, but if
  `rho`/`n_rot`/`diameter` are ALSO given, switches to COEFFICIENTS-ONLY
  (`C_T,axial`/`C_T,radial`/`C_T,tangential`/`C_Q`) and DROPS the
  dimensional N/N.m values entirely - this was an explicit
  confidentiality requirement from the user (don't show raw loads when
  coefficients are requested).
  - `C_T,* = force / (rho * n_rot^2 * diameter^4)` - SAME equation for
    all three force components (not separately "thrust vs. torque
    coefficient" for the tangential bar - that bar is still a force, not
    a torque).
  - `C_Q = torque / (rho * n_rot^2 * diameter^5)` - one extra factor of
    diameter since torque carries an extra length dimension.
- **Time domain / phase-locked / harmonics** (needs an "inst"
  multi-frame file, not an "average" 1-frame one - each method checks
  `n_frames` and raises a clear error otherwise):
  - `plot_time_trace()` - raw per-strip force vs time.
  - `phase_lock()` / `plot_vs_angle()` - bins frames by rotor azimuth
    (needs `rpm`, set on `StripForces.__init__`) and averages across
    however many revolutions are present. `polar=True/False` - Cartesian
    mode reproduces one subplot's data-curve content from the user's
    reference PDF (`Strip_PhasedLocked.pdf`), NOT the "model" comparison
    curve (no analytic model to overlay) or the full multi-panel layout.
  - `harmonics()` / `plot_harmonics()` - FFT at harmonics of the ROTOR'S
    OWN rotation frequency (1P, 2P, ... - per-blade, rotating frame).
    This is `|F_n(r)|`, Hanson's method's actual loading-harmonic input.
    Converting to an observer's actual blade-passage-frequency tones (at
    `n_blades` times this frequency) is Hanson's model's own downstream
    step, NOT computed here.
    - `return_phase=True` also returns each harmonic's phase - needed
      before this can actually feed Hanson's model (interference between
      radial stations' contributions depends on relative phase, not just
      magnitude).
    - `peak_azimuth()` - converts phase into the azimuth where each
      harmonic's contribution peaks - a physical, checkable claim (e.g.
      does the 1P peak line up with a known disturbance's actual
      position, like a strut). **Caveat, confirmed on synthetic data**:
      meaningless at harmonics whose magnitude is near the noise floor -
      don't read phase off a bar that's basically zero.
    - `reconstruct_from_harmonics()` - rebuilds an azimuth curve from
      magnitude+phase, for overlaying against `phase_lock()`'s real
      empirical curve as a validation/"how many harmonics actually
      matter" check.
    - `save_harmonics()` - the actual Hanson-ready output artifact (one
      self-contained HDF5: radius, chord, harmonic, magnitude, phase).
  - **All of the above validated against a SYNTHETIC signal with known,
    exact harmonic content** (`5 + 2*cos(2*pi*1*f_rot*t) + 1*cos(2*pi*3*f_rot*t)`,
    5 exact revolutions, no leakage by construction) - `harmonics()`
    recovered `2.0`/`1.0` N exactly (other harmonics at floating-point
    noise), `phase_lock()` matched the analytic curve to within 0.9%
    (expected bin-averaging smoothing, not an error), phase came back
    ~0° for the pure-cosine terms as expected. **NOT yet validated
    against real transient data** - no real multi-frame `.snc`
    conversion has been available locally this whole session (the only
    real converted file has just 2 frames). This is the single most
    important thing to actually run once real HPC data is accessible.

## `Cf` unsteadiness (`FrictionLines.cf(stat=...)`)

Added `stat='mean'/'rms'/'raw_rms'` to `cf()`, forwarded through
`cf_at_radii()`/`plot_cf_radii()`/`friction_lines()`. Computed on the
SCALAR Cf itself (magnitude or a signed component) frame-by-frame, THEN
reduced - NOT by reducing the wall-shear vector first (those differ,
since extracting magnitude/component is nonlinear; the MEAN case didn't
need this distinction since mean commutes with the linear
`tau = F - (F.n)n` projection, which is why `wall_shear(frame=None)`'s
existing behavior needed no change).
**Validated exactly** (~1e-15 relative error) against a synthetic
oscillating-force case with a known closed-form RMS answer.
**Not yet run against real data** - same limitation as above (need real
multi-frame data). When testing, expect a real result to look like a
dense dark field (real data has millions of surfels); the synthetic test
plots currently in the visible test folder deliberately look sparse and
noisy (only 5,000 random points, tiny domain) - that's a sample-size
artifact of the validation-only synthetic case, not a real result, and
was explicitly flagged as such to the user (don't mistake it for one).

## Known open risk: blade sweep (colleague's automotive cooling fan case)

Not yet tested against any swept-blade data. Assessment so far (logical,
not yet empirically checked):

- **Robust to sweep, no fix needed**: `SNCReader` conversion,
  `surface_split()` (normal-based, purely local), radius computation
  (`_radius()`, true distance from rotation axis), `StripForces`'s ENTIRE
  pipeline (thrust/torque/harmonics - built on the rotation axis, never
  on chord direction), `friction_lines()`'s raw scatter plot (no
  "one radius band = one chord station" assumption, just plots raw
  span/chord directly - would just look visually sheared, not wrong).
- **At risk under sweep**: anything using a THIN RADIUS BAND as a proxy
  for "one aerodynamic station's full chord, LE to TE" -
  `cf_at_radii()`/`plot_cf_radii()` (mean OR rms), `separation_line()`,
  `migration_line()`, `StripForces`'s `n_chord_bins`. The precise
  mechanism: sweep means a station's LE and TE genuinely sit at
  DIFFERENT true radii (that's what sweep geometrically is), so a
  constant-radius band either misses the LE/TE (if `tol` is small) or
  mixes in neighboring stations (if `tol` is widened enough to catch
  them) - same SYMPTOM as the two-blade-mixing bug, different CAUSE.
- **Proposed fix, not yet built** (needs real swept data to develop
  against): replace the constant-radius band with a band that follows
  the blade's actual swept reference line - either (a) a direct
  correction if the sweep angle is known analytically (shear the
  coordinate system before binning), or (b) numerically estimated from
  the mesh itself if it's not (track how each thin span-slice's chord
  centroid shifts, fit a reference line, band by distance from that
  curve instead of raw radius). Recommended NOT to build either blind -
  get one real file from the colleague first, check how much the
  local chord-axis range actually deviates from expectation, THEN decide
  which fix (or whether any fix) is actually needed.

## Pending / next steps (in likely priority order)

1. **Convert a real multi-frame (transient) `.snc` file** on the HPC -
   this unblocks real-data validation for: `StripForces`'s whole
   time-domain/phase-lock/harmonics suite, `Cf` RMS/unsteadiness. This is
   the biggest real gap right now - everything time-domain has only been
   validated against synthetic data with known answers, which proves the
   MATH is right but not that it looks/behaves sensibly on this actual
   rotor.
2. Get a real file from the colleague's swept-blade case, check the
   sweep risk empirically (see above) before trusting any
   `cf_at_radii()`-family result on it.
3. Possible future methods discussed but NOT built (only if the user
   wants them): proper streamline integration (`matplotlib.streamplot`)
   for the skin-friction topology instead of inferring it from the
   quiver; cross-validating `separation_line()` against
   `critical_points()` (separation lines are topologically required to
   emanate from saddle points - a free consistency check); sectional
   `c_l`/`c_d` distributions from the strip forces + local chord/`q_ref`;
   blade geometry (chord/thickness/twist) extraction for Hanson's model's
   non-loading inputs; flap/lag bending moments about a hinge (needs a
   hinge-offset input not currently part of the class).

## File/data conventions specific to this project's own validation case

- Real converted force file used for most of this session's earlier
  validation: `forces_rotor.h5`, built from `2f_SMF_forces_rotor.snc`
  via `SNCReader.to_h5(face_name='/Rotor::Default-Segment', surface_split=True)`.
  Only 2 frames - NOT a real transient case, just enough to sanity-check
  static/mean-case tools.
  **Note**: this file predates the `NX`→`Normal_X` rename and will need
  reconverting before `FrictionLines` can read it again.
- **Now also**: a real, ~400-frame transient case, `6e-5-6000rpm` (an
  isolated rotor in hover), converted via the same `forces` pipeline -
  this is the file at the center of the OPEN INVESTIGATION at the top of
  this document. Images referenced there live under
  `/Users/jmrendona/OneDrive - USherbrooke/PhD/rotor-alone/6e-5-6000rpm/images/`
  (note: different OneDrive root path, WITH spaces around the dash,
  than this repo's own `OneDrive-USherbrooke` working directory - both
  exist on this machine, don't confuse them).
- `r_tip=0.125` m, `rho_ref=1.22523` kg/m³, `rpm=6000` for this case
  (both the 2-frame and the 400-frame one - same underlying rotor).
- `span_min=0.02` isolates one blade half; `reverse_chord=True` needed
  to match the trusted `Cf`/`Cp` LE-at-x/c=0 convention.
