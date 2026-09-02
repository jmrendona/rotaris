import numpy as np
from converters.forces_strip import ForcesCSVConverter
from converters.span_2_radius import SpanConverter
from bladeprocessor.blades_postproc import BladePostProcessor
from bladeprocessor.surface_field import SurfaceField, SurfaceFieldComparator
from bladeprocessor.friction_lines import FrictionLines
from bladeprocessor.surface_variable import SurfaceVariable
from bladeprocessor.strip_forces import StripForces

# ------------- Convertors ------------- #

#forces = ForcesCSVConverter(
#    file_path='/scratch/renj3003/rotor-alone/6e-5_6000rpm/data/forces_strip',
#    file_pattern='Force-Graph-*.csv',
#    dt=0.000056
#)

#forces.convert('forces_strips.h5')

# timesteps = np.arange(0,46)

# for ts in timesteps:
# 	span_to_radius = SpanConverter(
#     	input_path=f'/scratch/renj3003/rotor-alone/6e-5_6000rpm-transitional/data/friction_lines/x-force/t{ts:02d}',
#     	output_path=f'/scratch/renj3003/rotor-alone/6e-5_6000rpm-transitional/data/friction_lines/x-force/t{ts:02d}/{ts:02d}t_x-force_radius.h5',
#     	variable_col='Value (Up)[ForcePerArea:newton/m^2]',
#     	span_col='Position (Up)[Length:m]',
#     	chord_length=0.025,
#     	resolution=0.025/100,
#         coordinate_system='cartesian',
#     	surface_split=False
# 	)

# 	span_to_radius.convert()

# ------------- Post processing ------------- #

#blade_cp = BladePostProcessor('/scratch/renj3003/rotor-alone/15e-6_6000rpm/data/cp/pstat_radius.h5', rpm = 6000, pref = 101325, rho_ref = 1.204)

#blade_cp.plot_radii(var_name = 'pressure', idx_list = [100, 200, 300, 400], mode = 'cp')

case_2025 = SurfaceField(
	"/storage/renj3003/rotor-alone/UdeS_Case/6e-5_6000rpm/data/cp/pstatic_cartesian.h5",
	var_name = 'pstatic',
	r_tip = 0.125,
	c_ref = 0.025
)

case_2025T = SurfaceField(
	"/storage/renj3003/rotor-alone/UdeS_Case/6e-5_6000rpm-transitional/data/cp/pstatic_cartesian.h5",
	var_name = 'pstatic',
	r_tip = 0.125
)

case_2026 = SurfaceField(
	"/storage/renj3003/rotor-alone/UdeS_Case/15e-6_6000rpm/data/cp/pstatic_cartesian.h5",
	var_name = 'pstatic',
	r_tip = 0.125
)

case_2025.plot_contour(normalize=False, cbar_label='Static Pressure [Pa]', levels=np.linspace(98000, 101800, 100),savepath='/storage/renj3003/rotor-alone/UdeS_Case/6e-5_6000rpm/images/pstatic/avg_pstatic.png')

comparator = SurfaceFieldComparator({'2025': case_2025, '2025-T': case_2025T, '2026': case_2026})

comparator.plot_cases(cbar_label='Static Pressure [Pa]', levels=np.linspace(98000, 101800, 100), savepath='/storage/renj3003/rotor-alone/UdeS_Case/Comparison/images/cp/avg_pstatic_comparison.png')

# ------------- Wall shear / friction lines ------------- #
#
# Input: a SNCReader.to_h5(..., surface_split=True) file - the forces
# branch, NEVER the pf2ens/pressure one (see README.md, "Splitting into
# upper/lower surface"). Built from EVERY raw frame, NEVER from a
# PowerFLOW-pre-averaged .snc (e.g. an "Avg"/"SMF"-style file) - confirmed
# to produce non-physical Cf on a real case, see README.md's "Input:
# every raw frame, never a PowerFLOW-pre-averaged .snc". Let
# FrictionLines do the time-averaging itself (frame=None below).
# rho_ref/rpm set the LOCAL Cf normalization
# (q_ref = 0.5*rho_ref*(omega*r)^2, matching BladePostProcessor.compute_cf()
# above - see README.md's "Equations" section for the full derivation).

#fl = FrictionLines(
#    '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/forces/forces_rotor.h5',
#    r_tip=0.125,
#    rho_ref=1.22523,
#    rpm=6000,
#)

# Dimensional wall shear vector (tau = F - (F.n)n), no rho_ref/rpm needed:
#tau = fl.wall_shear(surface='Upper', frame=None)  # frame=None -> average over every frame in the file

# Cf magnitude and signed chordwise/spanwise components, one frame or the average:
#cf_mag = fl.cf(surface='Upper', frame=None, component=None)
#cf_chordwise_frame0 = fl.cf(surface='Upper', frame=0, component='chordwise')

# Cf vs local x/c at several radii, one plot per call - instantaneous and
# average. span_min isolates one blade (REQUIRED in practice - without it,
# a radius band mixes both blades' chord ranges and produces a spurious
# double peak, see README.md); reverse_chord fixes which end is the
# leading vs. trailing edge (no automatic detection - check per case, Cf
# should peak sharply near x/c=0 and decay toward x/c=1; if it's flipped,
# set reverse_chord=True - see README.md's "Two bugs found and fixed"):
#fl.plot_cf_radii(
#    radii=[0.045, 0.072, 0.100, 0.117, 0.122],
#    surface='Upper', frame=None, component=None, span_min=0.02, reverse_chord=True,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/cf_radii_avg.png',
#)
#fl.plot_cf_radii(
#    radii=[0.045, 0.072, 0.100, 0.117, 0.122],
#    surface='Upper', frame=0, component='chordwise', span_min=0.02, reverse_chord=True,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/cf_radii_chordwise_frame0.png',
#)

# Cf unsteadiness (RMS fluctuation about the mean - see README.md, "Cf
# unsteadiness"): flags transition/wandering separation lines/moving
# vortex cores that the mean Cf field alone can miss.
#fl.plot_cf_radii(
#    radii=[0.045, 0.072, 0.100, 0.117, 0.122],
#    surface='Upper', frame=None, stat='rms', span_min=0.02, reverse_chord=True,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/cf_rms_radii_avg.png',
#)
#fl.friction_lines(
#    surface='Upper', frame=None, stat='rms', span_min=0.02,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/cf_rms_map.png',
#)

# Friction lines (Upper+Lower stacked by default) - span_min isolates one
# blade half on a two-bladed rotor centered at span=0 (see the method's
# docstring - there's no reliable automatic hub cutoff, pass what's right
# for this case's mesh):
#fl.friction_lines(
#    frame=None, span_min=0.03,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/friction_lines_avg.png',
#)

# Separation/reattachment line (chordwise-Cf sign crossings) - restricted
# to one blade section via span_min/span_max like everything else here;
# reverse_chord must match what plot_cf_radii()/cf_at_radii() needed on
# this case (see README.md, "Separation/reattachment line"):
#sep_points = fl.separation_line(surface='Upper', frame=None, span_min=0.02, reverse_chord=True)
#fl.save_separation_line(sep_points, '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/cf/separation_line.txt')

# Overlaid directly on friction_lines() (separation in red, reattachment in cyan):
#fl.friction_lines(
#    surface='Upper', frame=None, span_min=0.02, show_separation_line=True,
#    separation_line_kwargs={'reverse_chord': True},
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/friction_lines_separation.png',
#)

# Spanwise migration-reversal line (spanwise-Cf sign crossings - a
# DIFFERENT physical phenomenon from separation/reattachment above, see
# README.md, "Spanwise migration-reversal line"). edge_crop (default
# 0.05) excludes crossings too close to the LE/TE - real LE noise on
# this case, found and fixed this way after two amplitude-based filter
# attempts backfired (see the README section and migration_line()'s own
# docstring for the full story):
#mig_points = fl.migration_line(surface='Upper', frame=None, span_min=0.02, reverse_chord=True)
#fl.save_migration_line(mig_points, '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/cf/migration_line.txt')

#fl.friction_lines(
#    surface='Upper', frame=None, span_min=0.02, show_migration_line=True,
#    migration_line_kwargs={'reverse_chord': True},
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/friction_lines_migration.png',
#)

# Vortex-footprint critical points (node/saddle/focus - see README.md,
# "Vortex-footprint critical points"; 'focus' = actual vortex core, e.g.
# a leading-edge or corner/horseshoe vortex, not just an ordinary
# separation/reattachment feature). No reverse_chord - works in raw
# physical (span, chord) coordinates, not x/c:
#crit_points = fl.critical_points(surface='Upper', frame=None, span_min=0.02)
#fl.save_critical_points(crit_points, '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/cf/critical_points.txt')
#print('Poincare index N+F-S =', fl.poincare_index(crit_points))  # see README.md - NOT expected to be 2 on this open, cropped selection

# show_critical_points_index=True annotates the figure itself with N+F-S:
#fl.friction_lines(
#    surface='Upper', frame=None, span_min=0.02, show_critical_points=True, show_critical_points_index=True,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/friction_lines_critical_points.png',
#)

# ------------- Any surface variable at radii (Cp, y+, RMS, ...) ------------- #
#
# Input: a convert_snc_to_h5(..., surface_split=True) file - the pressure
# branch, for Cp (needs pf2ens's Static Pressure, see README.md, "2. Static
# Pressure"). Works just as well against a SNCReader.to_h5() forces-branch
# file for any variable stored there instead (Skin_Friction, y+ if
# present, etc.) - see README.md, "Any surface variable at radii".

#sv = SurfaceVariable(
#    '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/pressure/pressure_rotor.h5',
#    r_tip=0.125,
#    rho_ref=1.22523,
#    rpm=6000,
#    pref=101325,
#)

# Raw access to any stored variable - instantaneous, mean, or rms/raw_rms:
#yplus_mean = sv.variable('y+', surface='Upper', frame=None, stat='mean')
#yplus_frame0 = sv.variable('y+', surface='Upper', frame=0)  # stat ignored once frame is set

# Cp (same LOCAL q_ref normalization as FrictionLines.cf() - see README.md's
# "Equations" section), one frame, the average, or its RMS fluctuation:
#cp_mean = sv.cp(surface='Upper', frame=None, stat='mean')
#cp_frame0 = sv.cp(surface='Upper', frame=0)
#cp_rms = sv.cp(surface='Upper', frame=None, stat='rms')

# Cp vs local x/c at several radii, BOTH surfaces in one plot - span_min
# isolates one blade half (see friction_lines() above for why), and
# reverse_chord fixes which end is the leading vs. trailing edge (no
# automatic detection - check per case, see the method's docstring):
#sv.plot_cp_radii(
#    radii=[0.045, 0.072, 0.100, 0.117, 0.122],
#    frame=None, stat='mean', span_min=0.03, reverse_chord=True,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cp/cp_radii_avg.png',
#)
#sv.plot_cp_radii(
#    radii=[0.045, 0.072, 0.100, 0.117, 0.122],
#    frame=0, span_min=0.03, reverse_chord=True,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cp/cp_radii_frame0.png',
#)

# ------------- Any surface variable over the whole blade + case comparison ------------- #
#
# Generalizes friction_lines() (above) to any scalar field, and
# to_common_grid()/field() lets a SurfaceVariable slot into the existing
# SurfaceField/SurfaceFieldComparator machinery for cross-case deltas -
# see README.md, "Whole-blade surface plot" / "Cross-case comparison".

# Whole-blade -Cp scatter, both surfaces:
#sv.plot_variable_surface(
#    lambda s: -sv.cp(surface=s, stat='mean'),
#    cbar_label='-Cp', span_min=0.03,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cp/cp_surface_avg.png',
#)

# Cp resampled onto a common (r/R, x/c) grid, compared against a second
# case with the same geometry (c_ref must be passed explicitly - see
# README.md for why):
#sv_2025 = SurfaceVariable(
#    '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/pressure/pressure_rotor.h5',
#    r_tip=0.125, rho_ref=1.22523, rpm=6000, pref=101325,
#)
#sv_2026 = SurfaceVariable(
#    '/storage/renj3003/rotor-alone/15e-6_6000rpm/data/pressure/pressure_rotor.h5',
#    r_tip=0.125, rho_ref=1.22523, rpm=6000, pref=101325,
#)

#field_2025 = sv_2025.field(lambda s: sv_2025.cp(surface=s, stat='mean'),
#                            var_name='Cp 2025', c_ref=0.025, span_min=0.03)
#field_2026 = sv_2026.field(lambda s: sv_2026.cp(surface=s, stat='mean'),
#                            var_name='Cp 2026', c_ref=0.025, span_min=0.03)

#comparator_sv = SurfaceFieldComparator({'2025': field_2025, '2026': field_2026})
#comparator_sv.plot_cases(cbar_label='Cp', savepath='/storage/renj3003/rotor-alone/Comparison/images/cp/cp_comparison.png')
#comparator_sv.plot_delta('2025', '2026', cbar_label='Cp delta', savepath='/storage/renj3003/rotor-alone/Comparison/images/cp/cp_delta.png')

# Pressure fluctuation p'(frame) = p(frame) - p_mean, one blade contour
# per frame - needs a multi-frame file to show a real signal (a
# single-frame file gives exactly 0 everywhere, since p(frame) == p_mean):
#for frame in range(sv.n_frames):
#    sv.plot_pressure_fluctuation(
#        frame, span_min=0.03,
#        savepath=f'/storage/renj3003/rotor-alone/6e-5_6000rpm/images/pfluct/p_fluct_frame{frame:03d}.png',
#    )

# Prms needs no new method - it's already variable(stat='rms'):
#sv.plot_variable_surface(
#    lambda s: sv.variable('static_pressure', surface=s, stat='rms'),
#    cbar_label='$P_{rms}$ [Pa]', span_min=0.03,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/pfluct/p_rms_surface.png',
#)

# Point time trace + Welch periodogram (wall pressure fluctuations at one
# location, given as % of r/R and x/c - see README.md, "Point time trace").
# Needs a real time axis: pass dt explicitly if this file has no usable
# Metadata/mid_s (see README.md for when that's populated):
#sv.plot_timetrace(
#    'static_pressure', span_pct=80, chord_pct=90, surface='Upper',
#    ylabel='Static pressure [Pa]', dt=0.000056,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/spectra/p_timetrace_80_90.png',
#)
#sv.plot_periodogram(
#    'static_pressure', span_pct=80, chord_pct=90, surface='Upper',
#    ylabel='PSD [Pa$^2$/Hz]', dt=0.000056,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/spectra/p_periodogram_80_90.png',
#)

# ------------- Strip forces (Hanson's method input) ------------- #
#
# Per-radial-strip, time-resolved axial/radial/tangential force, computed
# directly from a SNCReader.to_h5() file - replaces the manual PowerVIZ
# "Force Graph" CSV export (ForcesCSVConverter above). span_min isolates
# one blade (see README.md, "Strip forces" - same reason as everywhere
# else in this project). Check flip_axial/flip_tangential against what
# you expect physically before trusting the sign.

#sf = StripForces(
#    '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/forces/forces_rotor.h5',
#    r_tip=0.125,
#)

#result = sf.compute(span_min=0.02, n_span_bins=20)
#sf.save(result, '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/forces/strip_forces.h5', dt=0.000056)

#sf.plot_bar_forces(
#    result, show_totals=True,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/forces/strip_forces_bar.png',
#)

# Chordwise-subdivided (non-compact-chord case - see README.md):
#result_2d = sf.compute(span_min=0.02, n_span_bins=20, n_chord_bins=5)
#sf.save(result_2d, '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/forces/strip_forces_2d.h5', dt=0.000056)

# Integrated totals (thrust/torque/radial/tangential force, independent of
# strip binning - see README.md, "Integrated totals"). result['totals']
# is guaranteed consistent with the span_min/span_max compute() above
# used; total_loads() is the same thing as a standalone call:
#print('thrust [N]:', result['totals']['thrust'].mean())
#print('torque [N.m]:', result['totals']['torque'].mean())
#totals = sf.total_loads(span_min=0.02)  # standalone, no strip binning needed

# Thrust/torque coefficients (propeller convention, C_F = F/(rho*n_rot^2*D^4),
# C_Q = Q/(rho*n_rot^2*D^5) - see README.md, "Thrust/torque coefficients").
# n_rot is rev/s, NOT RPM:
#sf.plot_bar_forces(
#    result, show_totals=True, rho=1.22523, n_rot=6000 / 60, diameter=0.25,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/forces/strip_forces_bar_coeffs.png',
#)

# Physical radius instead of r/R on the x-axis:
#sf.plot_bar_forces(
#    result, show_totals=True, normalize_radius=False,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/forces/strip_forces_bar_radius.png',
#)

# ------------- Time domain / phase-locked / harmonics (Hanson's method) ------------- #
#
# Only meaningful on an "inst" (multi-frame/transient) file - see
# README.md, "Average vs. instantaneous cases". Needs rpm (set on
# StripForces itself, not compute()) for phase_lock()/harmonics().

#sf_inst = StripForces(
#    '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/forces/forces_rotor_transient.h5',
#    r_tip=0.125, rpm=6000,
#)
#result_inst = sf_inst.compute(span_min=0.02, n_span_bins=20)

# Raw per-strip time trace (see README.md, "Time trace"):
#sf_inst.plot_time_trace(
#    result_inst, dt=0.000056, component='axial', strips=[0, 4, 9, 14, 19],
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/forces/strip_time_trace.png',
#)

# Phase-locked (revolution-folded) force vs azimuth (see README.md,
# "Phase-locked (revolution-folded) forces"):
#phase_locked = sf_inst.phase_lock(result_inst, dt=0.000056, n_azimuth_bins=72)
#sf_inst.plot_vs_angle(
#    phase_locked, component='axial', strips=[0, 4, 9, 14, 19],
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/forces/strip_vs_angle.png',
#)

# Harmonics of the rotation frequency - Hanson's method's actual |F_n(r)|
# input (see README.md, "Harmonics (Hanson's method's actual input)"):
#h = sf_inst.harmonics(result_inst, dt=0.000056, component='axial', n_harmonics=17)
#sf_inst.plot_harmonics(
#    h, strips=[0, 4, 9, 14, 19],
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/forces/strip_harmonics.png',
#)

# With phase (needed before actually handing this to Hanson's model, or
# to check a harmonic's peak azimuth against a known physical cause -
# see README.md, "Phase"):
#h_phase = sf_inst.harmonics(result_inst, dt=0.000056, component='axial', n_harmonics=17, return_phase=True)
#sf_inst.plot_harmonics(
#    h_phase, strips=[0, 4, 9, 14, 19], show_phase=True,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/forces/strip_harmonics_phase.png',
#)
#peak_deg = sf_inst.peak_azimuth(h_phase)  # (n_harmonics, n_span_bins)

# Reconstruction check against phase_lock()'s own empirical curve:
#phase_locked = sf_inst.phase_lock(result_inst, dt=0.000056, n_azimuth_bins=72)
#az, recon = sf_inst.reconstruct_from_harmonics(h_phase, azimuth_deg=phase_locked['azimuth_deg'])

# Hanson-model-ready output file (radius/chord/harmonic/magnitude/phase,
# self-contained, no need for this class or the .snc-derived file again):
#sf_inst.save_harmonics(h_phase, '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/forces/strip_harmonics.h5')