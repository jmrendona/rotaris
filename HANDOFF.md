# Handoff: rotaris post-processing toolkit — session context

This file summarizes an extended collaborative session building out the
`rotaris` toolkit, so a fresh Claude Code session (e.g. on the HPC, where
the real multi-frame `.snc` data lives) can pick up with full context
instead of starting cold. `README.md` is the canonical reference for
*how to use* everything below — this file is about *why things are the
way they are*, what's been validated against real vs. synthetic data,
and what's still open.

## OPEN INVESTIGATION — confirmed root cause, fix needs to happen on the local machine

**Status**: the ORIGINAL investigation (Cf magnitude apparently growing
25x with frame index) is RESOLVED - see "RESOLVED: Cf-magnitude-growth
investigation" below. Investigating it surfaced a SEPARATE, more serious,
CONFIRMED bug (not yet fixed): `Surface_X/Y/Z-Force` is written to every
converted `.h5` in the GLOBAL (lab, non-rotating) reference frame, while
`Geometry/{Upper,Lower}` (positions, normals - and therefore the
`chord_axis`/`span_axis`/`radial`/`tangential` directions everything
downstream defines from them) is in the LOCAL (LRF, blade-co-rotating)
frame. Nothing currently rotates the force vector into the LRF before
splitting it into chordwise/spanwise (`FrictionLines`) or
radial/tangential (`StripForces`) components. **This needs an actual fix
- do that first, on the local machine, before trusting any chordwise/
spanwise/radial/tangential-component result (torque, harmonics,
phase-lock, separation/migration lines, critical points) from ANY
`.snc` conversion, old or new.**

### Why this was missed until now

Confirmed directly from the `.h5` schema: `Geometry/*` datasets have NO
frame axis at all (shape `(n_points,)`, IDENTICAL across every frame -
this is `SNCReader.to_h5()`'s own documented behavior, "the raw .snc file
carries no frame axis for it, only for measurements"), while `Data/*`
(`Surface_X/Y/Z-Force`, `Skin_Friction`, `Static_Pressure`) has shape
`(n_frames, n_points)`. All of `FrictionLines`'s and `StripForces`'s
prior "validated on real data" claims elsewhere in this file were done
either on the frame-averaged mean, or on the only real multi-frame file
available at the time (2 frames, ~2 degrees apart - see below, that
tiny gap turns out to be exactly why). A rotation-frame mismatch this
small over 2 degrees is invisible; it only becomes obvious once you
compare frames spread across a real fraction of a revolution - which
this session finally had, in `forces_out.h5`
(`/scratch/jmrendon/Rotor-alone/6e-5_6000rpm/forces_out.h5`, 829 frames,
~4.6 revolutions, already converted, no raw `.snc` needed for any of the
checks below).

### Evidence (all against `forces_out.h5`, `span_min=0.02`, one blade
half, `rho_ref=1.22523`, `rpm=6000`, `span_axis=0`, `chord_axis=2`)

1. **Net in-plane wall-shear angle sweeps with frame index.** Summing
   `tau[chord_axis]` and `tau[span_axis]` (`tau = F - (F.n)n`) over every
   selected surfel per frame, then taking `atan2(span_sum, chord_sum)`,
   traces out a SMOOTH, nearly-monotonic sweep across one whole
   revolution (frames 0→176, step 4): Upper goes 168.7°→-148.8° (span
   317.5°, average rate -1.80°/frame); Lower goes 173.9°→-155.9° (span
   329.8°, average rate -1.87°/frame). A vector that's genuinely steady
   (or only mildly modulated) in the blade's own frame should NOT sweep
   360° once per revolution if it's actually being expressed in that
   frame - it should if it's still in the global frame instead.
2. **De-rotating by the known mechanical rate collapses the sweep.**
   Subtracting `+2.0°/frame` from the unwrapped angle (undoing the
   apparent `-2°/frame` drift) drops the sweep span from 317.5°→109.3°
   (Upper) and 329.8°→89.8° (Lower), with residual slope falling from
   ~-1.8/-1.9°/frame to +0.67/+0.27°/frame. Subtracting the WRONG sign
   (-2.0°/frame) makes it worse instead (span balloons to ~670-680°) -
   this sign-sensitivity rules out coincidence.
3. **The `.snc` file itself independently confirms the exact rate.**
   `SNCReader` already reads (but never uses) `lrf_axis_origin`,
   `lrf_axis_direction`, and `lrf_constant_angular_vel_mag`
   (`self.lrf_angular_vel_lattice`) - the file's OWN recorded LRF
   angular velocity. The raw NetCDF ALSO carries (currently NOT read by
   `SNCReader` at all): `start_time`/`end_time` (real per-frame
   timestamps, shape `(nsets,)` = `(n_frames,)` - `SNCReader.to_h5()`
   currently throws these away and writes a fake `Metadata/frame_index =
   np.arange(n_frames)` instead), plus `lrf_has_constant_angular_vel`,
   `lrf_initial_angular_rotation`, `lrf_initial_n_revolutions`. Reading
   these directly off a real file (`f50_SMF_forces_rotor.snc`, a small
   2-frame raw dump for this same case) and computing
   `lrf_constant_angular_vel_mag * (start_time[1] - start_time[0])`
   (no extra unit conversion needed - `LatticeTime` scale is exactly
   `1.0` for this case) gives **exactly -2.001 degrees** for that file's
   single frame-to-frame gap - matching the independently-measured
   empirical drift (~-1.8 to -2.4°/frame, from a completely different
   file/method) almost exactly. `lrf_has_constant_angular_vel=1` (true)
   and `lrf_initial_angular_rotation=0.0` for this case.
4. **Corroborating (secondary) signal**: a chordwise/spanwise
   cross-correlation check at two frame pairs 90° apart, both away from
   the low-Cf trough (frame 40 vs 85, and the identical relative pair 4
   revolutions later, 760 vs 805) shows a real cross-component signature
   (e.g. Upper: `corr(chordwise40,chordwise85)=0.982`,
   `corr(spanwise40,chordwise85)=0.847`) that reproduces almost exactly
   4 revolutions apart - consistent with a rotation-angle-tied effect,
   though on its own this was ambiguous (an earlier attempt at 1-vs-45
   was confounded by frame 1 sitting near the Cf trough, i.e. near-zero
   signal - don't reuse that specific pair).

### The exact fix formula (derived, not yet implemented)

```
angle(frame) = lrf_initial_angular_rotation
             + lrf_constant_angular_vel_mag * (start_time[frame] - start_time[0])
```
(radians; `LatticeTime` scale was exactly `1.0` for this case, so no
further unit conversion was needed here - CONFIRM this holds in general
before hardcoding it, rather than assuming). Rotate `Surface_X/Y/Z-Force`
about `lrf_axis_direction` by `-angle(frame)` (sign TBD by direct
empirical check against the measured drift direction above) before any
chordwise/spanwise/radial/tangential decomposition. `Skin_Friction` and
`Static_Pressure` are SCALARS (confirmed via `diagnose_snc.py`'s section
2/3 output - no vector/frame ambiguity for those), so this only concerns
`Surface_X/Y/Z-Force`.

**Recommended fix location**: centrally, inside `SNCReader.to_h5()` /
`_write_surfel_group()` (`converters/snc_reader.py`) - decode
`start_time`, `lrf_initial_angular_rotation`, `lrf_has_constant_angular_vel`
in `_decode_metadata()`, rotate `Surface_X/Y/Z-Force` into the LRF there
before writing, and write the REAL per-frame time (not
`np.arange(n_frames)`) into `Metadata` too - so every downstream
consumer (`FrictionLines`, `StripForces`) gets already-correct data with
no changes needed on their end, and gets real timing for free (useful
independently for `StripForces`'s harmonics/FFT, which currently has to
assume uniform frame spacing). Confirm `lrf_has_constant_angular_vel` is
true before applying this formula as-is - if any case has it false, the
rotation isn't a simple constant rate and needs different handling.

### What's corrupted vs. safe

- **Corrupted**: `FrictionLines.cf(component='chordwise'/'spanwise')`,
  `separation_line()`, `migration_line()`, `critical_points()` (needs the
  full in-plane vector field). Their "validated on the real converted
  rotor case... found REAL, physically coherent structure" claim
  elsewhere in this file should be treated as UNRELIABLE until re-run
  after the fix.
- **Likely safe**: Cf MAGNITUDE (`component=None`, what
  `plot_cf_radii()` uses - the file that started this whole
  investigation). This blade is nearly flat, so its normal is dominated
  by the axis-aligned (frame-invariant) thickness direction, making
  `F.n` and hence `|tau|` only weakly sensitive to this rotation
  mismatch - consistent with the clean, tight magnitude periodicity
  already confirmed (frame 720 ≈ frame 0, frame 800 ≈ frame 80, see
  below).
- **Not yet directly tested, but almost certainly ALSO corrupted by the
  identical mechanism**: `StripForces`'s `radial`/`tangential`
  decomposition (built the same way - a basis vector from static LRF
  geometry, dotted against the same global-frame `Force`) - and
  therefore torque, `harmonics()`, `phase_lock()`, `plot_time_trace()`,
  everything time-resolved. `axial` (thrust) should be safe, since the
  rotation axis itself is frame-invariant under its own rotation -
  consistent with HANDOFF's existing thrust validation looking correct
  while torque was flagged as suspicious from the start. **Run the same
  angle-tracking test (net radial+tangential vector angle vs. frame)
  against `StripForces` before trusting any of its time-domain output.**

### RESOLVED: Cf-magnitude-growth investigation

Original symptom: `plot_cf_radii()`'s Cf magnitude appeared to grow
smoothly/monotonically with frame index then saturate (peak Cf ~0.044 at
frame 0, ~0.23 at frame 10, ~0.83 at frame 40, ~1.1 at frame 80).
`diagnose_snc.py` (repo root) confirmed `_LargeRecordNetcdfFile` reads
match plain `scipy.io.netcdf_file` bit-for-bit, frame-by-frame, on two
real files (a 121-frame/186 GB file for a different case, and this
case's own small raw dump) - the reader itself is exonerated, this was
never a `_LargeRecordNetcdfFile` bug.

Extending the check across all 829 frames of `forces_out.h5` (not just
0/10/40/80) showed the "growth" is actually PERIODIC, not monotonic:
frame 720 (=frame 0 + exactly 4 revolutions at ~2°/frame) reproduces
frame 0's Cf stats almost exactly (mean 0.0086 vs 0.0082 Upper, 0.0070
vs 0.0068 Lower), and frame 800 reproduces frame 80 just as tightly
(0.1222 vs 0.1221 Upper). The original 0/10/40/80 sample just happened
to land on the rising limb of one cycle, starting right near a periodic
trough - not unbounded growth. Real, periodic, once-per-revolution
near-wall shear variation (plausible for a hovering rotor encountering
its own wake/tip vortex each revolution) - not a bug, not a slow
"still-converging" transient.

`diagnose_snc.py` usage, if needed again: `python diagnose_snc.py
/path/to/case.snc` - opens via plain scipy AND `SNCReader` side by side,
prints per-frame magnitude for both.

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

**Now also validated against two real files this session** (see OPEN
INVESTIGATION above, "RESOLVED: Cf-magnitude-growth investigation"):
bit-for-bit identical to plain scipy, frame-by-frame, on a 186 GB/
121-frame file (different case) AND this project's own
`6e-5-6000rpm` raw dump. The reader itself is exonerated - still
untested on the original multi-GB DNS case that motivated it (needs the
HPC), but no longer an open question for this project's own files.
**IMPORTANT - separate, still-open issue**: `SNCReader.to_h5()` writes
`Surface_X/Y/Z-Force` in the wrong reference frame (global instead of
LRF) - a real, CONFIRMED bug, but a physics/frame bug, not a byte-reading
bug like the two above. See OPEN INVESTIGATION for the fix.

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

**CAVEAT (added this session, see OPEN INVESTIGATION)**: this validation
predates the discovery that `Surface_X/Y/Z-Force` is in the wrong
reference frame, and all three of these tools consume the
chordwise/spanwise-split wall-shear VECTOR (not just its magnitude) -
exactly what that bug corrupts. Treat this validation as UNRELIABLE and
re-run it after the frame fix lands, before trusting it again.

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
  similarly confirmed ~half the user's known total. **CAVEAT (this
  session, see OPEN INVESTIGATION)**: this validation was done on the
  only real file available at the time (2 frames, ~2 degrees apart) -
  too small a gap to reveal the confirmed `Surface_X/Y/Z-Force`
  global-vs-LRF frame bug. `axial` (thrust) is derived from the rotation
  axis itself, which is frame-invariant, so it's likely still fine
  (consistent with it validating cleanly) - but `radial`/`tangential`
  (and therefore torque) use the exact same mechanism as
  `FrictionLines`'s corrupted chordwise/spanwise split (a basis vector
  from static LRF geometry, dotted against the still-global `Force`) and
  are almost certainly ALSO wrong. Re-validate after the frame fix.
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
    real converted file has just 2 frames). **Also now blocked on the
    OPEN INVESTIGATION's frame-of-reference fix** - `radial`/
    `tangential` (hence torque, and every harmonic/phase-lock/time-trace
    output below, all radial/tangential-based) are almost certainly
    corrupted by the same global-vs-LRF `Surface_X/Y/Z-Force` bug until
    that's fixed - don't spend real validation effort here until then,
    it would need redoing anyway. This is the single most
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

1. **Fix the `Surface_X/Y/Z-Force` global-vs-LRF reference frame bug** -
   see OPEN INVESTIGATION at the top. This blocks trusting
   `separation_line()`/`migration_line()`/`critical_points()`, and
   almost certainly `StripForces`'s torque/radial/tangential/harmonics/
   phase-lock/time-trace, entirely. Highest priority - everything else
   in this list either depends on it or is lower-impact.
2. After the fix: re-validate `separation_line()`/`migration_line()`/
   `critical_points()` and `StripForces`'s full time-domain suite
   against real data (`forces_out.h5`, 829 frames, already converted -
   see File/data conventions below) - both were flagged this session as
   needing re-validation, not just re-running once.
3. Get a real file from the colleague's swept-blade case, check the
   sweep risk empirically (see above) before trusting any
   `cf_at_radii()`-family result on it.
4. Possible future methods discussed but NOT built (only if the user
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
- **Now also**: a real, 829-frame (~4.6 revolution) transient case,
  `6e-5-6000rpm` (an isolated rotor in hover) -
  `/scratch/jmrendon/Rotor-alone/6e-5_6000rpm/forces_out.h5` on the HPC,
  already converted, this is the file the whole OPEN INVESTIGATION above
  (both the resolved Cf-periodicity question and the still-open
  reference-frame bug) was diagnosed against. **The raw `.snc` source
  this was converted from no longer exists** (deleted from scratch to
  save space) - not needed for the frame-bug fix itself (the already-
  converted `.h5` has everything required), but if it's ever needed
  again it only exists on the user's other computer (~3+ hour transfer).
  A small (2-frame) raw `.snc` for this same case DOES still exist on
  the HPC scratch (`f50_SMF_forces_rotor.snc`, despite the misleading
  "f50" name - it only has 2 frames) - this is what was used to pull the
  exact `start_time`/`lrf_initial_angular_rotation`/etc. metadata for
  the fix formula above; useful as a quick real-data reference without
  needing the full 829-frame conversion. Images referenced earlier in
  this document for the original Cf-growth symptom live under
  `/Users/jmrendona/OneDrive - USherbrooke/PhD/rotor-alone/6e-5-6000rpm/images/`
  (note: different OneDrive root path, WITH spaces around the dash,
  than this repo's own `OneDrive-USherbrooke` working directory - both
  exist on this machine, don't confuse them).
- `r_tip=0.125` m, `rho_ref=1.22523` kg/m³, `rpm=6000` for this case
  (the 2-frame, 829-frame, and any other conversion of it - same
  underlying rotor).
- `span_min=0.02` isolates one blade half; `reverse_chord=True` needed
  to match the trusted `Cf`/`Cp` LE-at-x/c=0 convention.
