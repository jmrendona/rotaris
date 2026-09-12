import os
import glob
import numpy as np
from converters.span_2_radius import SpanConverter
from bladeprocessor.blades_postproc import BladePostProcessor
from bladeprocessor.surface_field import SurfaceField, SurfaceFieldComparator
from bladeprocessor.friction_lines import FrictionLines
from bladeprocessor.surface_variable import SurfaceVariable
from bladeprocessor.strip_forces import StripForces
from bladeprocessor.tip_vortex_tracking import TipVortexPhaseAverage
from bladeprocessor.convergence import plot_cumulative_stats
from bladeprocessor.convergence import plot_cumulative_mean
from bladeprocessor.convergence import plot_autocorrelation_windows

# ------------- Post processing ------------- #

#blade_cp = BladePostProcessor('/scratch/renj3003/rotor-alone/15e-6_6000rpm/data/cp/pstat_radius.h5', rpm = 6000, pref = 101325, rho_ref = 1.204)

#blade_cp.plot_radii(var_name = 'pressure', idx_list = [100, 200, 300, 400], mode = 'cp')

# case_2025 = SurfaceField(
# 	"/storage/renj3003/rotor-alone/UdeS_Case/6e-5_6000rpm/data/cp/pstatic_cartesian.h5",
# 	var_name = 'pstatic',
# 	r_tip = 0.125,
# 	c_ref = 0.025
# )

# case_2025T = SurfaceField(
# 	"/storage/renj3003/rotor-alone/UdeS_Case/6e-5_6000rpm-transitional/data/cp/pstatic_cartesian.h5",
# 	var_name = 'pstatic',
# 	r_tip = 0.125
# )

# case_2026 = SurfaceField(
# 	"/storage/renj3003/rotor-alone/UdeS_Case/15e-6_6000rpm/data/cp/pstatic_cartesian.h5",
# 	var_name = 'pstatic',
# 	r_tip = 0.125
# )

# case_2025.plot_contour(normalize=False, cbar_label='Static Pressure [Pa]', levels=np.linspace(98000, 101800, 100),savepath='/storage/renj3003/rotor-alone/UdeS_Case/6e-5_6000rpm/images/pstatic/avg_pstatic.png')

# comparator = SurfaceFieldComparator({'2025': case_2025, '2025-T': case_2025T, '2026': case_2026})

# comparator.plot_cases(cbar_label='Static Pressure [Pa]', levels=np.linspace(98000, 101800, 100), savepath='/storage/renj3003/rotor-alone/UdeS_Case/Comparison/images/cp/avg_pstatic_comparison.png')

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

master_path = '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm'
inst_force_file = '2025_forces_rotor.h5'
avg_force_file = '2025_avg_forces_rotor.h5'
inst_pressure_file = ''
avg_pressure_file = '2025_avg_pressure_rotor.h5'
case = '2025'

print(40*'-')
print('Opening FrictionLines file: ', os.path.join(master_path, inst_force_file))
# span_min=0.02 here (not just on every call below) - for a whole-rotor
# case with no separate blade parts to select via face_name at
# conversion time, every single call below already passes span_min=0.02
# anyway (to isolate one blade - see the note further down), so cropping
# at load time means the (potentially huge) force field is only ever
# read for the surviving ~half of the points, not the whole rotor - see
# FrictionLines.__init__'s span_min/span_max docstring. Fixed a real OOM
# on a ~660 GB case this way.
fl = FrictionLines(
   os.path.join(master_path, inst_force_file),
   r_tip=0.125,
   rho_ref=1.22523,
   rpm=6000,
   span_min=0.02,
)

# # Dimensional wall shear vector (tau = F - (F.n)n), no rho_ref/rpm needed:
# tau = fl.wall_shear(surface='Upper', frame=None)  # frame=None -> average over every frame in the file

# Cf magnitude and signed chordwise/spanwise components, one frame or the average:
cf_mag = fl.cf(surface='Upper', frame=None, component=None)
print(40*'-')
print('Average Cf magnitude: ', np.mean(cf_mag))
# cf_chordwise_frame0 = fl.cf(surface='Upper', frame=0, component='chordwise')

for frame in range(0,fl.n_frames,15):
	cf_mag = fl.cf(surface='Upper', frame=frame, component=None)
	print(40*'-')
	print(f'Cf magnitude: {np.mean(cf_mag)} at frame {frame:03d}')

# Cf vs local x/c at several radii, one plot per call - instantaneous and
# average. span_min isolates one blade (REQUIRED in practice - without it,
# a radius band mixes both blades' chord ranges and produces a spurious
# double peak, see README.md); reverse_chord fixes which end is the
# leading vs. trailing edge (no automatic detection - check per case, Cf
# should peak sharply near x/c=0 and decay toward x/c=1; if it's flipped,
# set reverse_chord=True - see README.md's "Two bugs found and fixed"):

print(40*'-')
print('Plotting Cf vs x/c at several radii, Upper surface, average over all frames')
print(40*'-')
print('Plotting Cf vs x/c magnitude')
fl.plot_cf_radii(
   radii=[0.045, 0.072, 0.100, 0.117, 0.122],
   frame=None, component=None, span_min=0.02, reverse_chord=True,
   savepath=os.path.join(master_path, f'images/cf/avg/cf_radii_mag_avg_{case}.png'),
)
print(40*'-')
print('Plotting Cf vs x/c chordwise component')
fl.plot_cf_radii(
   radii=[0.045, 0.072, 0.100, 0.117, 0.122],
   frame=None, component='chordwise', span_min=0.02, reverse_chord=True,
   savepath=os.path.join(master_path, f'images/cf/avg/cf_radii_chordwise_avg_{case}.png'),
)
print(40*'-')
print('Plotting Cf vs x/c spanwise component')
fl.plot_cf_radii(
   radii=[0.045, 0.072, 0.100, 0.117, 0.122],
   frame=None, component='spanwise', span_min=0.02, reverse_chord=True,
   savepath=os.path.join(master_path, f'images/cf/avg/cf_radii_spanwise_avg_{case}.png'),
)

for frame in range(0,fl.n_frames,15):
	print(40*'-')
	print(f'Plotting Cf vs x/c at several radii, Upper surface, average for frame {frame:03d}')
	print(40*'-')
	print('Plotting Cf vs x/c magnitude')
	fl.plot_cf_radii(
	radii=[0.045, 0.072, 0.100, 0.117, 0.122],
	frame=frame, component=None, span_min=0.02, reverse_chord=True,
	savepath=os.path.join(master_path, f'images/cf/inst/cf_radii_mag_frame{frame:03d}_{case}.png'),
	)
	print(40*'-')
	print('Plotting Cf vs x/c chordwise component')
	fl.plot_cf_radii(
	radii=[0.045, 0.072, 0.100, 0.117, 0.122],
	frame=frame, component='chordwise', span_min=0.02, reverse_chord=True,
	savepath=os.path.join(master_path, f'images/cf/inst/cf_radii_chordwise_frame{frame:03d}_{case}.png'),
	)
	print(40*'-')
	print('Plotting Cf vs x/c spanwise component')
	fl.plot_cf_radii(
	radii=[0.045, 0.072, 0.100, 0.117, 0.122],
	frame=frame, component='spanwise', span_min=0.02, reverse_chord=True,
	savepath=os.path.join(master_path, f'images/cf/inst/cf_radii_spanwise_frame{frame:03d}_{case}.png'),
	)

# Cf unsteadiness (RMS fluctuation about the mean - see README.md, "Cf
# unsteadiness"): flags transition/wandering separation lines/moving
# vortex cores that the mean Cf field alone can miss.
print(40*'-')
print('Plotting Cf RMS vs x/c at several radii, Upper surface, average over all frames')
print(40*'-')
print('Plotting Cf RMS vs x/c magnitude')
fl.plot_cf_radii(
   radii=[0.045, 0.072, 0.100, 0.117, 0.122],
   surface='Upper', frame=None, stat='rms', span_min=0.02, reverse_chord=True,
   savepath=os.path.join(master_path, f'images/cf/cf_rms_radii_avg_{case}.png'),
)
print(40*'-')
print('Plotting Cf RMS vs x/c color map')
fl.friction_lines(
   surface='Upper', frame=None, stat='rms', span_min=0.02,
   savepath=os.path.join(master_path, f'images/cf/cf_rms_map_{case}.png'),
)

# Friction lines (Upper+Lower stacked by default) - span_min isolates one
# blade half on a two-bladed rotor centered at span=0 (see the method's
# docstring - there's no reliable automatic hub cutoff, pass what's right
# for this case's mesh):
print(40*'-')
print('Plotting Friction Lines, Upper surface, average over all frames')
fl.friction_lines(
   frame=None, span_min=0.02, surface='Upper',
   savepath=os.path.join(master_path, f'images/cf/avg/friction_lines_avg_{case}.png'),
)

for frame in range(0,fl.n_frames,15):
	print(40*'-')
	print(f'Plotting Friction Lines, Upper surface, for frame {frame:03d}')
	fl.friction_lines(
	   frame=frame, span_min=0.02, surface='Upper',
	   savepath=os.path.join(master_path, f'images/cf/inst/friction_lines_frame{frame:03d}_{case}.png'),
	)

# Separation/reattachment line (chordwise-Cf sign crossings) - restricted
# to one blade section via span_min/span_max like everything else here;
# reverse_chord must match what plot_cf_radii()/cf_at_radii() needed on
# this case (see README.md, "Separation/reattachment line"):
#sep_points = fl.separation_line(surface='Upper', frame=None, span_min=0.02, reverse_chord=True)
#fl.save_separation_line(sep_points, os.path.join(master_path, 'data/cf/separation_line.txt'))

# Overlaid directly on friction_lines() (separation in red, reattachment in cyan):
print(40*'-')
print('Plotting Friction Lines with separation/reattachment line, Upper surface, average over all frames')
fl.friction_lines(
   surface='Upper', frame=None, span_min=0.02, show_separation_line=True,
   separation_line_kwargs={'reverse_chord': True},
   savepath=os.path.join(master_path, f'images/cf/avg/friction_lines_separation_{case}.png'),
)

for frame in range(0,fl.n_frames,15):
	print(40*'-')
	print(f'Plotting Friction Lines with separation/reattachment line, Upper surface, for frame {frame:03d}')
	fl.friction_lines(
	   surface='Upper', frame=frame, span_min=0.02, show_separation_line=True,
	   separation_line_kwargs={'reverse_chord': True},
	   savepath=os.path.join(master_path, f'images/cf/inst/friction_lines_separation_frame{frame:03d}_{case}.png'),
	)

# Spanwise migration-reversal line (spanwise-Cf sign crossings - a
# DIFFERENT physical phenomenon from separation/reattachment above, see
# README.md, "Spanwise migration-reversal line"). edge_crop (default
# 0.05) excludes crossings too close to the LE/TE - real LE noise on
# this case, found and fixed this way after two amplitude-based filter
# attempts backfired (see the README section and migration_line()'s own
# docstring for the full story):
#mig_points = fl.migration_line(surface='Upper', frame=None, span_min=0.02, reverse_chord=True)
#fl.save_migration_line(mig_points, os.path.join(master_path, 'data/cf/migration_line.txt'))

#fl.friction_lines(
#    surface='Upper', frame=None, span_min=0.02, show_migration_line=True,
#    migration_line_kwargs={'reverse_chord': True},
#    savepath=os.path.join(master_path, 'images/cf/friction_lines_migration.png'),
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
print(40*'-')
print('Plotting Friction Lines with critical points, Upper surface, average over all frames')
fl.friction_lines(
   surface='Upper', frame=None, span_min=0.02, show_critical_points=True, show_critical_points_index=False,
   savepath=os.path.join(master_path, f'images/cf/avg/friction_lines_critical_points_{case}.png'),
)

for frame in range(0,fl.n_frames,15):
	print(40*'-')
	print(f'Plotting Friction Lines with critical points, Upper surface, for frame {frame:03d}')
	fl.friction_lines(
	surface='Upper', frame=frame, span_min=0.02, show_critical_points=True, show_critical_points_index=False,
	savepath=os.path.join(master_path, f'images/cf/inst/friction_lines_critical_points_frame{frame:03d}_{case}.png'),
	)

# Convergence checking: Cf phase portrait (see README.md, "Convergence
# checking: wall-shear/Cf phase portraits") - near-wall/viscous
# quantities converge MORE SLOWLY than integrated forces, so this needs
# checking separately from StripForces' phase portraits even if those
# already look converged:
print(40*'-')
print('Plotting Cf phase portrait (magnitude vs chordwise), Upper surface')
fl.plot_cf_phase_portrait(
   component_pair=(None, 'chordwise'), surface='Upper', span_min=0.02,
   savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_mag_chordwise_{case}.png'),
)

print(40*'-')
print('Plotting Cf phase portrait (magnitude vs spanwise), Upper surface')
fl.plot_cf_phase_portrait(
   component_pair=(None, 'spanwise'), surface='Upper', span_min=0.02,
   savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_mag_spanwise_{case}.png'),
)

print(40*'-')
print('Plotting Cf phase portrait (magnitude vs spanwise), Upper surface')
fl.plot_cf_phase_portrait(
   component_pair=('spanwise', 'chordwise'), surface='Upper', span_min=0.02,
   savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_spanwise_chordwise_{case}.png'),
)

print(40*'-')
print('Plotting per-strip Cf phase portraits (spanwise vs chordwise) - localizes convergence issues by span')
fl.plot_cf_phase_portrait_by_strip(
   component_pair=(None, 'chordwise'), surface='Upper', span_min=0.02, n_span_bins=10, strips=[2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_mag_chordwise_by_strip_{case}.png'),
)

print(40*'-')
print('Plotting per-strip Cf phase portraits (spanwise vs chordwise) - localizes convergence issues by span')
fl.plot_cf_phase_portrait_by_strip(
   component_pair=(None, 'spanwise'), surface='Upper', span_min=0.02, n_span_bins=10, strips=[2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_mag_spanwise_by_strip_{case}.png'),
)

print(40*'-')
print('Plotting per-strip Cf phase portraits (spanwise vs chordwise) - localizes convergence issues by span')
fl.plot_cf_phase_portrait_by_strip(
   component_pair=('spanwise', 'chordwise'), surface='Upper', span_min=0.02, n_span_bins=10, strips=[2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_spanwise_chordwise_by_strip_{case}.png'),
)

# ------------- Any surface variable at radii (Cp, y+, RMS, ...) ------------- #
#
# Input: a convert_snc_to_h5(..., surface_split=True) file - the pressure
# branch, for Cp (needs pf2ens's Static Pressure, see README.md, "2. Static
# Pressure"). Works just as well against a SNCReader.to_h5() forces-branch
# file for any variable stored there instead (Skin_Friction, y+ if
# present, etc.) - see README.md, "Any surface variable at radii".

print(40*'-')
print('Opening SurfaceVariable file: ', os.path.join(master_path, avg_pressure_file))
sv_pressure = SurfaceVariable(
   os.path.join(master_path, avg_pressure_file),
   r_tip=0.125,
   rho_ref=1.22523,
   rpm=6000,
   pref=101325,
)

print(40*'-')
print('Opening SurfaceVariable file: ', os.path.join(master_path, avg_force_file))
sv_forces = SurfaceVariable(
   os.path.join(master_path, avg_force_file),
   r_tip=0.125,
   rho_ref=1.22523,
   rpm=6000,
   pref=101325,
)

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
print(40*'-')
print('Plotting Cp vs x/c at several radii, average over all frames')
sv_pressure.plot_cp_radii(
   radii=[0.045, 0.072, 0.100, 0.117, 0.122],
   frame=None, stat='mean', span_min=0.02, reverse_chord=True,
   savepath=os.path.join(master_path, f'images/cp/cp_radii_avg_{case}.png'),
)
# sv_pressure.plot_cp_radii(
#    radii=[0.045, 0.072, 0.100, 0.117, 0.122],
#    frame=0, span_min=0.03, reverse_chord=True,
#    savepath=os.path.join(master_path, f'images/cp/cp_radii_frame0_{case}.png'),
# )

# ------------- Any surface variable over the whole blade + case comparison ------------- #
#
# Generalizes friction_lines() (above) to any scalar field, and
# to_common_grid()/field() lets a SurfaceVariable slot into the existing
# SurfaceField/SurfaceFieldComparator machinery for cross-case deltas -
# see README.md, "Whole-blade surface plot" / "Cross-case comparison".

# Whole-blade -Cp scatter, both surfaces:
print(40*'-')
print('Plotting -Cp surface scatter, average over all frames')
sv_pressure.plot_variable_surface(
   lambda s: -sv_pressure.cp(surface=s, stat='mean'),
   cbar_label='-Cp', span_min=0.02, surface='Upper',
   savepath=os.path.join(master_path, f'images/cp/cp_surface_avg_upper_{case}.png'),
)

print(40*'-')
print('Plotting Skin Friction surface scatter, average over all frames')
sv_forces.plot_variable_surface(
   lambda s: sv_forces.variable('Skin_Friction', surface=s, stat='mean'),
   cbar_label='Skin Friction [Pa]', span_min=0.02, surface='Upper',
   savepath=os.path.join(master_path, f'images/cf/avg/cf_surface_avg_upper_{case}.png'),
)


# Leading-edge stagnation point (potential-flow interaction with a
# downstream obstruction shifts it off the LE, toward whichever surface
# sees the higher effective incidence - see README.md, "Leading-edge
# stagnation point"). Sweeps span in bins, searching BOTH surfaces
# together (unlike everything else here, which is already split) for the
# local Cp maximum near x/c=0:
# points_stag = sv_pressure.stagnation_line(stat='mean', span_min=0.02)

# Compare the mean against a couple of individual frames - the "does it
# move frame to frame" question this was built for:
# sv_pressure.plot_stagnation_line(
#    {'mean': points_stag, 'frame 0': sv_pressure.stagnation_line(frame=0, span_min=0.02),
#     'frame 50': sv_pressure.stagnation_line(frame=50, span_min=0.03)},
#    savepath=os.path.join(master_path, 'images/cp/stagnation_vs_span.png'),
# )
#sv.save_stagnation_line(points_stag, os.path.join(master_path, 'data/cp/stagnation_mean.txt'))

# Or see it directly on the blade contour, jumping between the Upper/Lower
# subplots as it migrates sides - needs BOTH surfaces plotted:
# print(40*'-')
# print('Plotting -Cp surface scatter with stagnation line, average over all frames')
# sv_pressure.plot_variable_surface(
#    lambda s: -sv_pressure.cp(surface=s, stat='mean'),
#    cbar_label='-Cp', span_min=0.02, show_stagnation_line=True,
#    savepath=os.path.join(master_path, f'images/cp/cp_surface_with_stagnation_{case}.png'),
# )

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
#comparator_sv.plot_cases(cbar_label='Cp', savepath=os.path.join(master_path, 'images/cp/cp_comparison.png'))
#comparator_sv.plot_delta('2025', '2026', cbar_label='Cp delta', savepath=os.path.join(master_path, 'images/cp/cp_delta.png'))

# Pressure fluctuation p'(frame) = p(frame) - p_mean, one blade contour
# per frame - needs a multi-frame file to show a real signal (a
# single-frame file gives exactly 0 everywhere, since p(frame) == p_mean):
#for frame in range(sv_pressure.n_frames):
#    sv_pressure.plot_pressure_fluctuation(
#        frame, span_min=0.02,
#        savepath=os.path.join(master_path, f'images/pfluct/p_fluct_frame{frame:03d}.png'),
#    )

# Prms needs no new method - it's already variable(stat='rms'):
#sv_pressure.plot_variable_surface(
#    lambda s: sv_pressure.variable('static_pressure', surface=s, stat='rms'),
#    cbar_label='$P_{rms}$ [Pa]', span_min=0.03,
#    savepath=os.path.join(master_path, 'images/pfluct/p_rms_surface.png'),
#)

# Point time trace + Welch periodogram (wall pressure fluctuations at one
# location, given as % of r/R and x/c - see README.md, "Point time trace").
# Needs a real time axis: pass dt explicitly if this file has no usable
# Metadata/mid_s (see README.md for when that's populated):
#sv_pressure.plot_timetrace(
#    'static_pressure', span_pct=80, chord_pct=90, surface='Upper',
#    ylabel='Static pressure [Pa]', dt=0.000056,
#    savepath=os.path.join(master_path, 'images/spectra/p_timetrace_80_90.png'),
#)
#sv_pressure.plot_periodogram(
#    'static_pressure', span_pct=80, chord_pct=90, surface='Upper',
#    ylabel='PSD [Pa$^2$/Hz]', dt=0.000056,
#    savepath=os.path.join(master_path, 'images/spectra/p_periodogram_80_90.png'),
#)

# ------------- Strip forces (Hanson's method input) ------------- #
#
# Per-radial-strip, time-resolved axial/radial/tangential force, computed
# directly from a SNCReader.to_h5() file - replaces the manual PowerVIZ
# "Force Graph" CSV export (ForcesCSVConverter above). span_min isolates
# one blade (see README.md, "Strip forces" - same reason as everywhere
# else in this project). Check flip_axial/flip_tangential against what
# you expect physically before trusting the sign.

print(40*'-')
print('Opening StripForces file: ', os.path.join(master_path, inst_force_file))
# span_min=0.02 at load time (see FrictionLines' fl = ... above for why) -
# every compute()/total_loads() call below already passes span_min=0.02
# anyway to isolate one blade, so this crops the force field actually
# read off disk to the same subset, instead of loading the whole rotor.
sf_avg = StripForces(
   os.path.join(master_path, avg_force_file),
   r_tip=0.125,
   span_min=0.02,
)

print(40*'-')
print('Computing strip forces')
result = sf_avg.compute(span_min=0.02, n_span_bins=10)
#sf_avg.save(result, os.path.join(master_path, 'data/forces/strip_forces.h5'), dt=0.000056)

print(40*'-')
print('Plotting strip forces bar chart averageg over all frames')
sf_avg.plot_bar_forces(
   result, show_totals=False,
   savepath=os.path.join(master_path, f'images/forces/strip_forces_bar_avg_{case}.png'),
)

# # Chordwise-subdivided (non-compact-chord case - see README.md):
# #result_2d = sf.compute(span_min=0.02, n_span_bins=20, n_chord_bins=5)
# #sf.save(result_2d, os.path.join(master_path, 'data/forces/strip_forces_2d.h5'), dt=0.000056)

# Integrated totals (thrust/torque/radial/tangential force, independent of
# strip binning - see README.md, "Integrated totals"). result['totals']
# is guaranteed consistent with the span_min/span_max compute() above
# used; total_loads() is the same thing as a standalone call:
print(40*'-')
print('thrust [N]:', result['totals']['thrust'].mean())
print(40*'-')
print('torque [N.m]:', result['totals']['torque'].mean())
#totals = sf.total_loads(span_min=0.02)  # standalone, no strip binning needed

# Thrust/torque coefficients (propeller convention, C_F = F/(rho*n_rot^2*D^4),
# C_Q = Q/(rho*n_rot^2*D^5) - see README.md, "Thrust/torque coefficients").
# n_rot is rev/s, NOT RPM:
print(40*'-')
print('Plotting strip forces bar chart averageg over all frames with non-dimensional coefficients')
sf_avg.plot_bar_forces(
   result, show_totals=False, rho=1.22523, n_rot=6000 / 60, diameter=0.25,
   savepath=os.path.join(master_path, f'images/forces/strip_forces_bar_coeffs_avg_{case}.png'),
)

# Physical radius instead of r/R on the x-axis:
#sf.plot_bar_forces(
#    result, show_totals=True, normalize_radius=False,
#    savepath=os.path.join(master_path, 'images/forces/strip_forces_bar_radius.png'),
#)

# ------------- Time domain / phase-locked / harmonics (Hanson's method) ------------- #
#
# Only meaningful on an "inst" (multi-frame/transient) file - see
# README.md, "Average vs. instantaneous cases". Needs rpm (set on
# StripForces itself, not compute()) for phase_lock()/harmonics().

print(40*'-')
print('Opening StripForces file: ', os.path.join(master_path, inst_force_file))
# span_min=0.02 at load time - see sf_avg above. This is the big
# multi-frame/transient file, so this is the crop that actually matters
# for memory (the one that OOM-killed a real ~660 GB whole-rotor case
# before this parameter existed).
sf_inst = StripForces(
   os.path.join(master_path, inst_force_file),
   r_tip=0.125, rpm=6000,
   span_min=0.02,
)

print(40*'-')
print('Computing instantaneous strip forces')
result_inst = sf_inst.compute(span_min=0.02, n_span_bins=10)

# Raw per-strip time trace (see README.md, "Time trace"):
print(40*'-')
print('Plotting instantaneous strip forces time trace for the axial component')
sf_inst.plot_time_trace(
   result_inst, dt=0.000056, component='axial', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/strip_time_trace_axial_{case}.png'),
)

print(40*'-')
print('Plotting instantaneous strip forces time trace for the radial component')
sf_inst.plot_time_trace(
   result_inst, dt=0.000056, component='radial', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/strip_time_trace_radial_{case}.png'),
)

print(40*'-')
print('Plotting instantaneous strip forces time trace for the tangential component')
sf_inst.plot_time_trace(
   result_inst, dt=0.000056, component='tangential', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/strip_time_trace_tangential_{case}.png'),
)

# Phase-locked (revolution-folded) force vs azimuth (see README.md,
# "Phase-locked (revolution-folded) forces"):
print(40*'-')
print('Plotting phase-locked forces vs azimuth for the axial component')
phase_locked = sf_inst.phase_lock(result_inst, dt=0.000056, n_azimuth_bins=72)
sf_inst.plot_vs_angle(
   phase_locked, component='axial', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/strip_vs_angle_axial_{case}.png'),
)

print(40*'-')
print('Plotting phase-locked forces vs azimuth for the radial component')
sf_inst.plot_vs_angle(
   phase_locked, component='radial', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/strip_vs_angle_radial_{case}.png'),
)

print(40*'-')
print('Plotting phase-locked forces vs azimuth for the tangential component')
sf_inst.plot_vs_angle(
   phase_locked, component='tangential', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/strip_vs_angle_tangential_{case}.png'),
)

# Harmonics of the rotation frequency - Hanson's method's actual |F_n(r)|
# input (see README.md, "Harmonics (Hanson's method's actual input)"):
print(40*'-')
print('Plotting harmonics for the axial component')
h = sf_inst.harmonics(result_inst, dt=0.000056, component='axial', n_harmonics=17)
sf_inst.plot_harmonics(
   h, strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/strip_harmonics_axial_{case}.png'),
)

print(40*'-')
print('Plotting harmonics for the radial component')
h = sf_inst.harmonics(result_inst, dt=0.000056, component='radial', n_harmonics=17)
sf_inst.plot_harmonics(
   h, strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/strip_harmonics_radial_{case}.png'),
)

print(40*'-')
print('Plotting harmonics for the tangential component')
h = sf_inst.harmonics(result_inst, dt=0.000056, component='tangential', n_harmonics=17)
sf_inst.plot_harmonics(
   h, strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/strip_harmonics_tangential_{case}.png'),
)

# With phase (needed before actually handing this to Hanson's model, or
# to check a harmonic's peak azimuth against a known physical cause -
# see README.md, "Phase"):
print(40*'-')
print('Plotting harmonics with phase for the axial component')
h_phase = sf_inst.harmonics(result_inst, dt=0.000056, component='axial', n_harmonics=17, return_phase=True)
sf_inst.plot_harmonics(
   h_phase, strips=[0, 2, 4, 6, 8, 9], show_phase=True,
   savepath=os.path.join(master_path, f'images/forces/strip_harmonics_phase_axial_{case}.png'),
)

print(40*'-')
print('Computing peak azimuth of each harmonic for the axial component')
peak_deg = sf_inst.peak_azimuth(h_phase)  # (n_harmonics, n_span_bins)
print(40*'-')
print('Axial component peak azimuth of each harmonic (deg): ', peak_deg)

print(40*'-')
print('Plotting harmonics with phase for the radial component')
h_phase = sf_inst.harmonics(result_inst, dt=0.000056, component='radial', n_harmonics=17, return_phase=True)
sf_inst.plot_harmonics(
   h_phase, strips=[0, 2, 4, 6, 8, 9], show_phase=True,
   savepath=os.path.join(master_path, f'images/forces/strip_harmonics_phase_radial_{case}.png'),
)

print(40*'-')
print('Computing peak azimuth of each harmonic for the radial component')
peak_deg = sf_inst.peak_azimuth(h_phase)  # (n_harmonics, n_span_bins)
print(40*'-')
print('Radial component peak azimuth of each harmonic (deg): ', peak_deg)


print(40*'-')
print('Plotting harmonics with phase for the tangential component')
h_phase = sf_inst.harmonics(result_inst, dt=0.000056, component='tangential', n_harmonics=17, return_phase=True)
sf_inst.plot_harmonics(
   h_phase, strips=[0, 2, 4, 6, 8, 9], show_phase=True,
   savepath=os.path.join(master_path, f'images/forces/strip_harmonics_phase_tangential_{case}.png'),
)

print(40*'-')
print('Computing peak azimuth of each harmonic for the tangential component')
peak_deg = sf_inst.peak_azimuth(h_phase)  # (n_harmonics, n_span_bins)
print(40*'-')
print('Tangential component peak azimuth of each harmonic (deg): ', peak_deg)


# Reconstruction check against phase_lock()'s own empirical curve:
#phase_locked = sf_inst.phase_lock(result_inst, dt=0.000056, n_azimuth_bins=72)
#az, recon = sf_inst.reconstruct_from_harmonics(h_phase, azimuth_deg=phase_locked['azimuth_deg'])

# Hanson-model-ready output file (radius/chord/harmonic/magnitude/phase,
# self-contained, no need for this class or the .snc-derived file again):
#sf_inst.save_harmonics(h_phase, os.path.join(master_path, 'data/forces/strip_harmonics.h5'))

# ------------- Convergence checking: phase portraits (see README.md) ------------- #
#
# Fx-vs-Fy-style plots (one force component vs another, over time) - a
# CLOSED loop means the run has settled into periodic operation; a
# drifting/spiraling trajectory means it hasn't yet. Needs an "inst" file,
# same as the time-domain block above - a single already-averaged frame
# has no trajectory to trace. See README.md, "Convergence checking: phase
# portraits" for the full explanation.

print(40*'-')
print('Plotting whole-blade phase portrait (axial vs tangential)')
totals_inst = sf_inst.total_loads(span_min=0.02)  # standalone total, same span as result_inst above
sf_inst.plot_phase_portrait(
   totals_inst, component_pair=('axial', 'tangential'),
   savepath=os.path.join(master_path, f'images/forces/phase_portrait_axial_tangential_{case}.png'),
)

print(40*'-')
print('Plotting whole-blade phase portrait (axial vs radial)')
totals_inst = sf_inst.total_loads(span_min=0.02)  # standalone total, same span as result_inst above
sf_inst.plot_phase_portrait(
   totals_inst, component_pair=('axial', 'radial'),
   savepath=os.path.join(master_path, f'images/forces/phase_portrait_axial_radial_{case}.png'),
)

print(40*'-')
print('Plotting whole-blade phase portrait (radial vs tangential)')
totals_inst = sf_inst.total_loads(span_min=0.02)  # standalone total, same span as result_inst above
sf_inst.plot_phase_portrait(
   totals_inst, component_pair=('tangential', 'radial'),
   savepath=os.path.join(master_path, f'images/forces/phase_portrait_tangential_radial_{case}.png'),
)


print(40*'-')
print('Plotting per-strip phase portraits (axial vs radial) - localizes convergence issues by span')
sf_inst.plot_phase_portrait_by_strip(
   result_inst, component_pair=('axial', 'radial'), strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/phase_portrait_by_strip_axial_radial_{case}.png'),
)

print(40*'-')
print('Plotting per-strip phase portraits (axial vs tangential) - localizes convergence issues by span')
sf_inst.plot_phase_portrait_by_strip(
   result_inst, component_pair=('axial', 'tangential'), strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/phase_portrait_by_strip_axial_tangential_{case}.png'),
)

print(40*'-')
print('Plotting per-strip phase portraits (tangential vs radial) - localizes convergence issues by span')
sf_inst.plot_phase_portrait_by_strip(
   result_inst, component_pair=('tangential', 'radial'), strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/phase_portrait_by_strip_tangential_radial_{case}.png'),
)

# Convergence checking: cumulative (running) mean vs revolutions included
# (see README.md, "Convergence checking: running/cumulative mean") - a
# converged quantity's running mean flattens to a horizontal asymptote.
# Not tied to StripForces specifically - takes any plain 1D array:
print(40*'-')
print('Plotting cumulative mean of thrust vs revolutions included')
plot_cumulative_mean(
   totals_inst['thrust'], dt=0.000056, rpm=6000, ylabel='Thrust [N]',
   savepath=os.path.join(master_path, f'images/forces/thrust_cumulative_mean_{case}.png'),
)

print(40*'-')
print('Plotting cumulative mean of thrust vs revolutions included')
plot_cumulative_mean(
   totals_inst['torque'], dt=0.000056, rpm=6000, ylabel='Torque [Nm]',
   savepath=os.path.join(master_path, f'images/forces/torque_cumulative_mean_{case}.png'),
)

# Mean AND variance together (Pope's <U>/<u'^2> pair), synced to
# revolution boundaries - REQUIRED to see a clean asymptote on a signal
# with a real periodic component (see README.md, "Mean AND variance
# together, synced to revolution boundaries" - a plain per-frame running
# mean of such a signal shows a persistent ripple that this removes):
print(40*'-')
print('Plotting cumulative mean+variance of thrust, synced to revolution boundaries')
plot_cumulative_stats(
   totals_inst['thrust'], dt=0.000056, rpm=6000, sync_to_revolution=True, ylabel='Thrust [N]',
   savepath=os.path.join(master_path, f'images/forces/thrust_cumulative_stats_sync_{case}.png'),
)

print(40*'-')
print('Plotting cumulative mean+variance of thrust, synced to revolution boundaries')
plot_cumulative_stats(
   totals_inst['torque'], dt=0.000056, rpm=6000, sync_to_revolution=True, ylabel='Torque [Nm]',
   savepath=os.path.join(master_path, f'images/forces/torque_cumulative_stats_sync_{case}.png'),
)

# Convergence checking: autocorrelation comparison between independent
# windows (see README.md, "Convergence checking: autocorrelation" - NOT
# a single-window "is rho(s) even" check, which is guaranteed to pass
# trivially regardless of convergence - comparing INDEPENDENT windows is
# what's actually meaningful):

print(40*'-')
print('Plotting thrust autocorrelation, first half vs second half of the run')
plot_autocorrelation_windows(
   totals_inst['thrust'], n_windows=2, dt=0.000056, labels=['First half', 'Second half'],
   savepath=os.path.join(master_path, f'images/forces/thrust_autocorrelation_windows_{case}.png'),
)

print(40*'-')
print('Plotting torque autocorrelation, first half vs second half of the run')
plot_autocorrelation_windows(
   totals_inst['torque'], n_windows=2, dt=0.000056, labels=['First half', 'Second half'],
   savepath=os.path.join(master_path, f'images/forces/torque_autocorrelation_windows_{case}.png'),
)

# ------------- Tip-vortex tracking: phase-locked plane averaging (see README.md) ------------- #
#
# Extraction is a cluster job (needs pf2ens - see run_conversion.sh), not
# run here - ONE-SIDED inplane_range (not the two-sided default) so each
# plane has a single search zone, spanning the FULL 360deg (one-sided
# planes don't get the opposite azimuth for free the way a two-sided
# diametral plane does), and --angle-step matched EXACTLY to this case's
# own per-frame rotation angle (~2.0011 deg/frame for 6e-5_6000rpm - see
# HANDOFF.md's reference-frame investigation - not a re-derived or
# rounded value) so the relabeling below is exact, not approximate:
#
#   sbatch run_conversion.sh fnc-meridional-sweep SMR-VR8.fnc \
#       /storage/renj3003/rotor-alone/6e-5_6000rpm/data/fnc/tip_vortex_planes/ \
#       --angle-start 0 --angle-end 358 --angle-step 2.0011 \
#       --inplane-range 0 0.13 --variables vx,vy,vz \
#       --first 0 --last 199 --nc-stats nc_stats.txt
#
# (--nc-stats matters here beyond timing metadata: it's what lets
# TipVortexPhaseAverage auto-detect the rotation direction below, from
# the resulting files' own Metadata/lrf_position_rad.)

#plane_paths = sorted(glob.glob(
#    '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/fnc/tip_vortex_planes/plane_*deg.h5'
#))
#tva = TipVortexPhaseAverage(plane_paths, spacing_deg=2.0011)

# cylindrical=True (default) converts (vx, vy, vz) into (v_r, v_theta, v_z) -
# omega_rad_s is deliberately NOT passed here (see README.md, "Two things
# flagged as open" - whether pf2ens's velocity is absolute or relative to
# the LRF is not yet verified, unlike Surface_X/Y/Z-Force):
#result = tva.compute(['vx', 'vy', 'vz'], cylindrical=True)

# One age label's averaged field, on whatever radial range the planes
# actually cover (one-sided here - NOT mirrored into a full diameter,
# see README.md):
#tva.plot_age_label(
#    result, label=5, variable='v_r',
#    savepath=os.path.join(master_path, 'images/tip_vortex/age5_vr.png'),
#)

# Every age label in one pass, e.g. for an animation across "wake age":
#for label in range(tva.n_planes):
#    tva.plot_age_label(
#        result, label=label, variable='v_r',
#        savepath=os.path.join(master_path, f'images/tip_vortex/age{label}_vr.png'),
#    )