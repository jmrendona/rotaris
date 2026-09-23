import os
import glob
import argparse
import numpy as np
import yaml
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
from bladeprocessor.convergence import plot_cumulative_moments
from bladeprocessor.convergence import standard_error
from bladeprocessor.convergence import plot_integral_timescale
from bladeprocessor.convergence import required_averaging_time
from bladeprocessor.convergence import plot_cycle_correlation

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

# ------------- Case config (see cases/*.yaml) ------------- #
#
# Switching to a different simulation = pointing this at a different
# cases/*.yaml (via --case, or just leave it to the default below), not
# hunting through the rest of this file for every hardcoded r_tip/
# rho_ref/rpm/span_min/etc. NOT every single call below uses these - a
# handful deliberately pass a different span_min/radii/etc. for that one
# specific plot (a tighter/looser crop, an alternate radii list) - those
# stay as literals at their own call site on purpose, only the values
# that are genuinely the same everywhere are pulled from here. See
# cases/6e-5_6000rpm_HF.yaml's own header comment.
#
# Relative paths are resolved against THIS file's own directory, not
# whatever the current working directory happens to be when manager.py
# is run - so an existing submit script that just runs
# `python3 ~/rotaris/manager.py` with no --case at all keeps using the
# DEFAULT below unchanged; add --case only where you want to override it.
_parser = argparse.ArgumentParser()
_parser.add_argument('--case', default='cases/6e-5_6000rpm_HF.yaml',
                      help='Path to the case config YAML (see cases/*.yaml) - relative to '
                           'this file\'s own directory unless given as an absolute path.')
_args = _parser.parse_args()

CASE_CONFIG_PATH = _args.case
if not os.path.isabs(CASE_CONFIG_PATH):
	CASE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CASE_CONFIG_PATH)

with open(CASE_CONFIG_PATH) as _f:
	_cfg = yaml.safe_load(_f)

master_path = _cfg['master_path']
case = _cfg['case']

inst_force_file = _cfg['files']['inst_force']
avg_force_file = _cfg['files']['avg_force']
inst_pressure_file = _cfg['files']['inst_pressure']
avg_pressure_file = _cfg['files']['avg_pressure']

r_tip = _cfg['rotor']['r_tip']
rho_ref = _cfg['rotor']['rho_ref']
rpm = _cfg['rotor']['rpm']
pref = _cfg['rotor']['pref']

span_axis = _cfg['axes']['span']
chord_axis = _cfg['axes']['chord']
thickness_axis = _cfg['axes']['thickness']
validate_axes = not _cfg['axes']['skip_validation']

span_min = _cfg['crop']['span_min']

reverse_chord = _cfg['friction']['reverse_chord']
radii = _cfg['friction']['radii']

dt = _cfg['convergence']['dt']

blade_figsize = tuple(_cfg['blade_figsize']) if _cfg['blade_figsize'] else None

frame_loop_step = _cfg['frame_loop_step']

# Output folders under master_path/images - exist_ok=True leaves any folder
# (and everything already saved in it) untouched on re-runs, and makedirs
# creates the parents (cf, cp, forces, pfluct) implicitly.
for _sub in ('cf/avg', 'cf/inst', 'cp/avg', 'cp/inst', 'cp/rms', 'cp/convergence',
             'forces/convergence', 'forces/hanson', 'pfluct/spectra', 'tip_vortex'):
	os.makedirs(os.path.join(master_path, 'images', _sub), exist_ok=True)

os.makedirs(os.path.join(master_path, 'data', 'pfluct'), exist_ok=True)


# # ------------- Friction related post-processing ------------- #
# print(40*'-')
# print('Opening FrictionLines file: ', os.path.join(master_path, inst_force_file))
# # span_min=span_min here (not just on every call below) - for a whole-rotor
# # case with no separate blade parts to select via face_name at
# # conversion time, every single call below already passes span_min=span_min
# # anyway (to isolate one blade - see the note further down), so cropping
# # at load time means the (potentially huge) force field is only ever
# # read for the surviving ~half of the points, not the whole rotor - see
# # FrictionLines.__init__'s span_min/span_max docstring. Fixed a real OOM
# # on a ~660 GB case this way.
# fl = FrictionLines(
#    os.path.join(master_path, inst_force_file),
#    r_tip=r_tip,
#    rho_ref=rho_ref,
#    rpm=rpm,
#    span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis,
#    span_min=span_min, validate_axes=validate_axes
# )

# # # Dimensional wall shear vector (tau = F - (F.n)n), no rho_ref/rpm needed:
# # tau = fl.wall_shear(surface='Upper', frame=None)  # frame=None -> average over every frame in the file

# # Cf magnitude and signed chordwise/spanwise components, one frame or the average:
# cf_mag = fl.cf(surface='Upper', frame=None, component=None)
# print(40*'-')
# print('Average Cf magnitude: ', np.mean(cf_mag))
# # cf_chordwise_frame0 = fl.cf(surface='Upper', frame=0, component='chordwise')

# for frame in range(0,fl.n_frames,frame_loop_step):
# 	cf_mag = fl.cf(surface='Upper', frame=frame, component=None)
# 	print(40*'-')
# 	print(f'Cf magnitude: {np.mean(cf_mag)} at frame {frame:03d}')

# # Cf vs local x/c at several radii, one plot per call - instantaneous and
# # average. span_min isolates one blade (REQUIRED in practice - without it,
# # a radius band mixes both blades' chord ranges and produces a spurious
# # double peak, see README.md); reverse_chord fixes which end is the
# # leading vs. trailing edge (no automatic detection - check per case, Cf
# # should peak sharply near x/c=0 and decay toward x/c=1; if it's flipped,
# # set reverse_chord=reverse_chord - see README.md's "Two bugs found and fixed"):

# print(40*'-')
# print('Plotting Cf vs x/c at several radii, Upper surface, average over all frames')
# print(40*'-')
# print('Plotting Cf vs x/c magnitude')
# fl.plot_cf_radii(
#    radii=radii,#[0.045, 0.072, 0.100, 0.117, 0.122],
#    frame=None, component=None, span_min=span_min, reverse_chord=reverse_chord,
#    savepath=os.path.join(master_path, f'images/cf/avg/cf_radii_mag_avg_{case}.png'),
# )
# print(40*'-')
# print('Plotting Cf vs x/c chordwise component')
# fl.plot_cf_radii(
#    radii=radii,
#    frame=None, component='chordwise', span_min=span_min, reverse_chord=reverse_chord,
#    savepath=os.path.join(master_path, f'images/cf/avg/cf_radii_chordwise_avg_{case}.png'),
# )
# print(40*'-')
# print('Plotting Cf vs x/c spanwise component')
# fl.plot_cf_radii(
#    radii=radii,
#    frame=None, component='spanwise', span_min=span_min, reverse_chord=reverse_chord,
#    savepath=os.path.join(master_path, f'images/cf/avg/cf_radii_spanwise_avg_{case}.png'),
# )

# for frame in range(0,fl.n_frames,frame_loop_step):
# 	print(40*'-')
# 	print(f'Plotting Cf vs x/c at several radii, Upper surface, average for frame {frame:03d}')
# 	print(40*'-')
# 	print('Plotting Cf vs x/c magnitude')
# 	fl.plot_cf_radii(
# 	radii=radii,
# 	frame=frame, component=None, span_min=span_min, reverse_chord=reverse_chord,
# 	savepath=os.path.join(master_path, f'images/cf/inst/cf_radii_mag_frame{frame:03d}_{case}.png'),
# 	)
# 	print(40*'-')
# 	print('Plotting Cf vs x/c chordwise component')
# 	fl.plot_cf_radii(
# 	radii=radii,
# 	frame=frame, component='chordwise', span_min=span_min, reverse_chord=reverse_chord,
# 	savepath=os.path.join(master_path, f'images/cf/inst/cf_radii_chordwise_frame{frame:03d}_{case}.png'),
# 	)
# 	print(40*'-')
# 	print('Plotting Cf vs x/c spanwise component')
# 	fl.plot_cf_radii(
# 	radii=radii,
# 	frame=frame, component='spanwise', span_min=span_min, reverse_chord=reverse_chord,
# 	savepath=os.path.join(master_path, f'images/cf/inst/cf_radii_spanwise_frame{frame:03d}_{case}.png'),
# 	)

# # Cf unsteadiness (RMS fluctuation about the mean - see README.md, "Cf
# # unsteadiness"): flags transition/wandering separation lines/moving
# # vortex cores that the mean Cf field alone can miss.
# print(40*'-')
# print('Plotting Cf RMS vs x/c at several radii, Upper surface, average over all frames')
# print(40*'-')
# print('Plotting Cf RMS vs x/c magnitude')
# fl.plot_cf_radii(
#    radii=radii,
#    surface='Upper', frame=None, stat='rms', span_min=span_min, reverse_chord=reverse_chord,
#    savepath=os.path.join(master_path, f'images/cf/cf_rms_radii_avg_{case}.png'),
# )
# print(40*'-')
# print('Plotting Cf RMS vs x/c color map')
# fl.friction_lines(
#    surface='Upper', frame=None, stat='rms', span_min=span_min,
#    figsize=blade_figsize,
#    savepath=os.path.join(master_path, f'images/cf/cf_rms_map_{case}.png'),
# )

# # Friction lines (Upper+Lower stacked by default) - span_min isolates one
# # blade half on a two-bladed rotor centered at span=0 (see the method's
# # docstring - there's no reliable automatic hub cutoff, pass what's right
# # for this case's mesh):
# print(40*'-')
# print('Plotting Friction Lines, Upper surface, average over all frames')
# fl.friction_lines(
#    frame=None, span_min=span_min, surface='Upper',
#    figsize=blade_figsize,
#    savepath=os.path.join(master_path, f'images/cf/avg/friction_lines_avg_{case}.png'),
# )

# for frame in range(0,fl.n_frames,frame_loop_step):
# 	print(40*'-')
# 	print(f'Plotting Friction Lines, Upper surface, for frame {frame:03d}')
# 	fl.friction_lines(
# 	   frame=frame, span_min=span_min, surface='Upper',
# 	   figsize=blade_figsize,
# 	   savepath=os.path.join(master_path, f'images/cf/inst/friction_lines_frame{frame:03d}_{case}.png'),
# 	)

# # Separation/reattachment line (chordwise-Cf sign crossings) - restricted
# # to one blade section via span_min/span_max like everything else here;
# # reverse_chord must match what plot_cf_radii()/cf_at_radii() needed on
# # this case (see README.md, "Separation/reattachment line"):
# #sep_points = fl.separation_line(surface='Upper', frame=None, span_min=span_min, reverse_chord=reverse_chord)
# #fl.save_separation_line(sep_points, os.path.join(master_path, 'data/cf/separation_line.txt'))

# # Overlaid directly on friction_lines() (separation in red, reattachment in cyan):
# print(40*'-')
# print('Plotting Friction Lines with separation/reattachment line, Upper surface, average over all frames')
# fl.friction_lines(
#    surface='Upper', frame=None, span_min=span_min, show_separation_line=True,
#    separation_line_kwargs={'reverse_chord': reverse_chord},
#    figsize=blade_figsize,
#    savepath=os.path.join(master_path, f'images/cf/avg/friction_lines_separation_{case}.png'),
# )

# for frame in range(0,fl.n_frames,frame_loop_step):
# 	print(40*'-')
# 	print(f'Plotting Friction Lines with separation/reattachment line, Upper surface, for frame {frame:03d}')
# 	fl.friction_lines(
# 	   surface='Upper', frame=frame, span_min=span_min, show_separation_line=True,
# 	   separation_line_kwargs={'reverse_chord': reverse_chord},
# 	   figsize=blade_figsize,
# 	   savepath=os.path.join(master_path, f'images/cf/inst/friction_lines_separation_frame{frame:03d}_{case}.png'),
# 	)

# # Spanwise migration-reversal line (spanwise-Cf sign crossings - a
# # DIFFERENT physical phenomenon from separation/reattachment above, see
# # README.md, "Spanwise migration-reversal line"). edge_crop (default
# # 0.05) excludes crossings too close to the LE/TE - real LE noise on
# # this case, found and fixed this way after two amplitude-based filter
# # attempts backfired (see the README section and migration_line()'s own
# # docstring for the full story):
# #mig_points = fl.migration_line(surface='Upper', frame=None, span_min=span_min, reverse_chord=reverse_chord)
# #fl.save_migration_line(mig_points, os.path.join(master_path, 'data/cf/migration_line.txt'))

# #fl.friction_lines(
# #    surface='Upper', frame=None, span_min=span_min, show_migration_line=True,
# #    migration_line_kwargs={'reverse_chord': True},
# #    figsize=blade_figsize,
# #    savepath=os.path.join(master_path, 'images/cf/friction_lines_migration.png'),
# #)

# Vortex-footprint critical points (node/saddle/focus - see README.md,
# "Vortex-footprint critical points"; 'focus' = actual vortex core, e.g.
# a leading-edge or corner/horseshoe vortex, not just an ordinary
# separation/reattachment feature). No reverse_chord - works in raw
# physical (span, chord) coordinates, not x/c:
#crit_points = fl.critical_points(surface='Upper', frame=None, span_min=span_min)
#fl.save_critical_points(crit_points, '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/cf/critical_points.txt')
#print('Poincare index N+F-S =', fl.poincare_index(crit_points))  # see README.md - NOT expected to be 2 on this open, cropped selection

# show_critical_points_index=True annotates the figure itself with N+F-S:
# print(40*'-')
# print('Plotting Friction Lines with critical points, Upper surface, average over all frames')
# fl.friction_lines(
#    surface='Upper', frame=None, span_min=span_min, show_critical_points=True, show_critical_points_index=False,
#    figsize=blade_figsize,
#    savepath=os.path.join(master_path, f'images/cf/avg/friction_lines_critical_points_{case}.png'),
# )

# for frame in range(0,fl.n_frames,frame_loop_step):
# 	print(40*'-')
# 	print(f'Plotting Friction Lines with critical points, Upper surface, for frame {frame:03d}')
# 	fl.friction_lines(
# 	surface='Upper', frame=frame, span_min=span_min, show_critical_points=True, show_critical_points_index=False,
# 	figsize=blade_figsize,
# 	savepath=os.path.join(master_path, f'images/cf/inst/friction_lines_critical_points_frame{frame:03d}_{case}.png'),
# 	)

# # Convergence checking: Cf phase portrait (see README.md, "Convergence
# # checking: wall-shear/Cf phase portraits") - near-wall/viscous
# # quantities converge MORE SLOWLY than integrated forces, so this needs
# # checking separately from StripForces' phase portraits even if those
# # already look converged:
# print(40*'-')
# print('Plotting Cf phase portrait (magnitude vs chordwise), Upper surface')
# fl.plot_cf_phase_portrait(
#    component_pair=(None, 'chordwise'), surface='Upper', span_min=span_min,
#    savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_mag_chordwise_{case}.png'),
# )

# print(40*'-')
# print('Plotting Cf phase portrait (magnitude vs spanwise), Upper surface')
# fl.plot_cf_phase_portrait(
#    component_pair=(None, 'spanwise'), surface='Upper', span_min=span_min,
#    savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_mag_spanwise_{case}.png'),
# )

# print(40*'-')
# print('Plotting Cf phase portrait (magnitude vs spanwise), Upper surface')
# fl.plot_cf_phase_portrait(
#    component_pair=('spanwise', 'chordwise'), surface='Upper', span_min=span_min,
#    savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_spanwise_chordwise_{case}.png'),
# )

# print(40*'-')
# print('Plotting per-strip Cf phase portraits (spanwise vs chordwise) - localizes convergence issues by span')
# fl.plot_cf_phase_portrait_by_strip(
#    component_pair=(None, 'chordwise'), surface='Upper', span_min=span_min, n_span_bins=10, strips=[2, 4, 6, 8, 9],
#    savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_mag_chordwise_by_strip_{case}.png'),
# )

# print(40*'-')
# print('Plotting per-strip Cf phase portraits (spanwise vs chordwise) - localizes convergence issues by span')
# fl.plot_cf_phase_portrait_by_strip(
#    component_pair=(None, 'spanwise'), surface='Upper', span_min=span_min, n_span_bins=10, strips=[2, 4, 6, 8, 9],
#    savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_mag_spanwise_by_strip_{case}.png'),
# )

# print(40*'-')
# print('Plotting per-strip Cf phase portraits (spanwise vs chordwise) - localizes convergence issues by span')
# fl.plot_cf_phase_portrait_by_strip(
#    component_pair=('spanwise', 'chordwise'), surface='Upper', span_min=span_min, n_span_bins=10, strips=[2, 4, 6, 8, 9],
#    savepath=os.path.join(master_path, f'images/cf/cf_phase_portrait_spanwise_chordwise_by_strip_{case}.png'),
# )

# # Convergence checking: cumulative mean+variance and higher-order
# # moments (skewness/flatness) of Cf ITSELF - the same tools already
# # used for thrust/torque above (see "Convergence checking" further
# # down), fed with Cf's own spatial-mean-per-frame series instead. Near-
# # wall/viscous quantities are known to converge MORE SLOWLY than
# # integrated forces (see the Cf phase-portrait note above), so this is
# # worth checking even once thrust/torque already look converged. No
# # extra span cropping needed here - fl was already constructed with
# # span_min=span_min, so cf_time_series() only ever covers the one
# # blade half fl was built with; just reduce it to one scalar per frame:
# cf_mag_series = fl.cf_time_series(surface='Upper', component=None).mean(axis=1)
# cf_chordwise_series = fl.cf_time_series(surface='Upper', component='chordwise').mean(axis=1)
# cf_spanwise_series = fl.cf_time_series(surface='Upper', component='spanwise').mean(axis=1)

# print(40*'-')
# print('Plotting cumulative mean+variance of Cf magnitude')
# plot_cumulative_stats(
#    cf_mag_series, dt=dt, rpm=rpm, sync='none', ylabel='$C_f$ [-]',
#    savepath=os.path.join(master_path, f'images/cf/cf_mag_cumulative_stats_{case}.png'),
# )

# print(40*'-')
# print('Plotting cumulative mean+variance of chordwise Cf')
# plot_cumulative_stats(
#    cf_chordwise_series, dt=dt, rpm=rpm, sync='none', ylabel='$C_{f,chordwise}$ [-]',
#    savepath=os.path.join(master_path, f'images/cf/cf_chordwise_cumulative_stats_{case}.png'),
# )

# print(40*'-')
# print('Plotting cumulative mean+variance of spanwise Cf')
# plot_cumulative_stats(
#    cf_spanwise_series, dt=dt, rpm=rpm, sync='none', ylabel='$C_{f,spanwise}$ [-]',
#    savepath=os.path.join(master_path, f'images/cf/cf_spanwise_cumulative_stats_{case}.png'),
# )

# print(40*'-')
# print('Plotting cumulative skewness+flatness of Cf magnitude')
# plot_cumulative_moments(
#    cf_mag_series, dt=dt, rpm=rpm, sync='none', label='$C_f$',
#    savepath=os.path.join(master_path, f'images/cf/cf_mag_cumulative_moments_{case}.png'),
# )

# print(40*'-')
# print('Plotting cumulative skewness+flatness of chordwise Cf')
# plot_cumulative_moments(
#    cf_chordwise_series, dt=dt, rpm=rpm, sync='none', label='$C_{f,chordwise}$',
#    savepath=os.path.join(master_path, f'images/cf/cf_chordwise_cumulative_moments_{case}.png'),
# )

# print(40*'-')
# print('Plotting cumulative skewness+flatness of spanwise Cf')
# plot_cumulative_moments(
#    cf_spanwise_series, dt=dt, rpm=rpm, sync='none', label='$C_{f,spanwise}$',
#    savepath=os.path.join(master_path, f'images/cf/cf_spanwise_cumulative_moments_{case}.png'),
# )

# ------------- Any surface variable at radii (Cp, y+, RMS, ...) ------------- #
#
# Input: a convert_snc_to_h5(..., surface_split=True) file - the pressure
# branch, for Cp (needs pf2ens's Static Pressure, see README.md, "2. Static
# Pressure"). Works just as well against a SNCReader.to_h5() forces-branch
# file for any variable stored there instead (Skin_Friction, y+ if
# present, etc.) - see README.md, "Any surface variable at radii".

# print(40*'-')
# print('Opening SurfaceVariable file: ', os.path.join(master_path, avg_pressure_file))
# sv_pressure = SurfaceVariable(
#    os.path.join(master_path, avg_pressure_file),
#    r_tip=r_tip,
#    rho_ref=rho_ref,
#    rpm=rpm,
#    pref=pref,
#    span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis
# )

# print(40*'-')
# print('Opening SurfaceVariable file: ', os.path.join(master_path, avg_force_file))
# sv_forces = SurfaceVariable(
#    os.path.join(master_path, avg_force_file),
#    r_tip=r_tip,
#    rho_ref=rho_ref,
#    rpm=rpm,
#    pref=pref,
#    span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis
# )

# # Raw access to any stored variable - instantaneous, mean, or rms/raw_rms:
#yplus_mean = sv.variable('y+', surface='Upper', frame=None, stat='mean')
#yplus_frame0 = sv.variable('y+', surface='Upper', frame=0)  # stat ignored once frame is set

# # Cp (same LOCAL q_ref normalization as FrictionLines.cf() - see README.md's
# # "Equations" section), one frame, the average, or its RMS fluctuation:
#cp_mean = sv.cp(surface='Upper', frame=None, stat='mean')
#cp_frame0 = sv.cp(surface='Upper', frame=0)
#cp_rms = sv.cp(surface='Upper', frame=None, stat='rms')

# ------------- Convergence checking on pressure (see README.md) ------------- #

# "Convergence checking on pressure") - needs an INSTANTANEOUS (multi-
# frame) pressure file, never a PowerFLOW-pre-averaged one, same
# requirement as everywhere else in this project:
# print(40*'-')
# print('Opening SurfaceVariable file: ', os.path.join(master_path, inst_pressure_file))
# sv_pressure_inst = SurfaceVariable(
#   os.path.join(master_path, inst_pressure_file),
#   r_tip=r_tip, rho_ref=rho_ref, rpm=rpm, pref=pref,
#   span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis
# )

# # Spatial mean pressure/Cp per frame - same reduction
# # plot_cf_phase_portrait() uses on cf_time_series() - every convergence.py
# # function takes this plain 1D per-frame series, exactly like thrust/torque:
# p_series = sv_pressure_inst.variable_time_series('static_pressure', surface='Upper').mean(axis=1)[:-2]  # drop corrupted last frame(s)
# cp_series = sv_pressure_inst.cp_time_series(surface='Upper').mean(axis=1)[:-2]

# print(40*'-')
# print('Plotting cumulative mean+variance of pressure')
# plot_cumulative_stats(
#   p_series, dt=dt, rpm=rpm, sync='none', ylabel='Pressure [Pa]',
#   savepath=os.path.join(master_path, f'images/cp/convergence/pressure_cumulative_stats_{case}.png'),
# )

# print(40*'-')
# print('Plotting integral timescale / required averaging time for pressure')
# plot_integral_timescale(
#   p_series, dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
#   savepath=os.path.join(master_path, f'images/cp/convergence/pressure_integral_timescale_{case}.png'),
# )

# print(40*'-')
# print('Plotting cumulative mean of pressure vs revolutions included')
# plot_cumulative_mean(
#   p_series, dt=dt, rpm=rpm, ylabel='Pressure [Pa]',
#   savepath=os.path.join(master_path, f'images/cp/convergence/pressure_cumulative_mean_{case}.png'),
# )

# print(40*'-')
# print('Plotting cumulative skewness+flatness of pressure')
# plot_cumulative_moments(
#   p_series, dt=dt, rpm=rpm, sync='none', label='Pressure',
#   savepath=os.path.join(master_path, f'images/cp/convergence/pressure_cumulative_moments_{case}.png'),
# )

# print(40*'-')
# print('Plotting pressure autocorrelation, first half vs second half of the run')
# plot_autocorrelation_windows(
#   p_series, n_windows=2, dt=dt, labels=['First half', 'Second half'],
#   savepath=os.path.join(master_path, f'images/cp/convergence/pressure_autocorrelation_windows_{case}.png'),
# )

# # print(40*'-')
# # print('Plotting cycle-to-cycle correlation of pressure')
# # plot_cycle_correlation(
# #   p_series, dt=dt, rpm=rpm, period_deg=72,
# #   savepath=os.path.join(master_path, f'images/cp/convergence/pressure_cycle_correlation_{case}.png'),
# # )


# ------------- Radii cuts profiles for surface variables ------------- #

# Cp vs local x/c at several radii, BOTH surfaces in one plot - span_min
# isolates one blade half (see friction_lines() above for why), and
# reverse_chord fixes which end is the leading vs. trailing edge (no
# automatic detection - check per case, see the method's docstring):
# print(40*'-')
# print('Plotting Cp vs x/c at several radii, average over all frames')
# sv_pressure.plot_cp_radii(
#    radii=radii,
#    frame=None, stat='mean', span_min=span_min, reverse_chord=reverse_chord,
#    savepath=os.path.join(master_path, f'images/cp/avg/cp_radii_avg_{case}.png'),
# )

# for frame in range(0, sv_pressure_inst.n_frames, frame_loop_step):
#   print(40*'-')
#   print('Plotting Cp vs x/c at several radii, instatenous frame ', frame)
#   sv_pressure_inst.plot_cp_radii(
#     radii=radii,
#     frame=frame, span_min=span_min, reverse_chord=reverse_chord,
#     savepath=os.path.join(master_path, f'images/cp/inst/cp_radii_frame{frame:03d}_{case}.png'),
# )

# print(40*'-')
# print('Plotting Cp rms vs x/c at several radii, instatenous files')
# sv_pressure_inst.plot_cp_radii(
#    radii=radii,
#    frame=None, stat='rms', span_min=span_min, reverse_chord=reverse_chord,
#    savepath=os.path.join(master_path, f'images/cp/rms/cp_radii_rms_{case}.png'),
# )

# # ------------- Any surface variable over the whole blade ------------- #
# #
# # Generalizes friction_lines() (above) to any scalar field, and
# # to_common_grid()/field() lets a SurfaceVariable slot into the existing
# # SurfaceField/SurfaceFieldComparator machinery for cross-case deltas -
# # see README.md, "Whole-blade surface plot" / "Cross-case comparison".

# # Whole-blade -Cp scatter, both surfaces:
# print(40*'-')
# print('Plotting -Cp surface scatter, average over all frames')
# sv_pressure.plot_variable_surface(
#    lambda s: -sv_pressure.cp(surface=s, stat='mean'),
#    cbar_label='-Cp', span_min=span_min, surface='Upper',
#    figsize=blade_figsize,
#    savepath=os.path.join(master_path, f'images/cp/avg/cp_surface_avg_upper_{case}.png'),
# )

# for frame in range(0, sv_pressure_inst.n_frames, frame_loop_step):
#   print(40*'-')
#   print('Plotting -Cp surface scatter, instantaneous frame ', frame)
#   sv_pressure_inst.plot_variable_surface(
#     lambda s: -sv_pressure_inst.cp(surface=s, frame=frame),
#     cbar_label='-Cp', span_min=span_min, surface='Upper',
#     figsize=blade_figsize,
#     savepath=os.path.join(master_path, f'images/cp/inst/cp_surface_upper_frame{frame:03d}_{case}.png'),
#   )

# print(40*'-')
# print('Plotting Skin Friction surface scatter, average over all frames')
# sv_forces.plot_variable_surface(
#    lambda s: sv_forces.variable('Skin_Friction', surface=s, stat='mean'),
#    cbar_label='Skin Friction [Pa]', span_min=span_min, surface='Upper',
#    figsize=blade_figsize,
#    savepath=os.path.join(master_path, f'images/cf/avg/cf_surface_avg_upper_{case}.png'),
# )

# # ------------- Leading-edge stagnation point ------------- #
# #
# # Leading-edge stagnation point (potential-flow interaction with a
# # downstream obstruction shifts it off the LE, toward whichever surface
# # sees the higher effective incidence - see README.md, "Leading-edge
# # stagnation point"). Sweeps span in bins, searching BOTH surfaces
# # together (unlike everything else here, which is already split) for the
# # local Cp maximum near x/c=0:
# points_stag = sv_pressure.stagnation_line(stat='mean', span_min=span_min)

# # Compare the mean against a couple of individual frames - the "does it
# # move frame to frame" question this was built for:
# sv_pressure.plot_stagnation_line(
#    {'mean': points_stag, 'frame 0': sv_pressure.stagnation_line(frame=0, span_min=span_min),
#     'frame 50': sv_pressure.stagnation_line(frame=50, span_min=span_min)},
#    savepath=os.path.join(master_path, f'images/cp/stagnation_vs_span_{case}.png'),
# )
#sv.save_stagnation_line(points_stag, os.path.join(master_path, f'data/cp/stagnation_mean_{case}.txt'))

# # Or see it directly on the blade contour, jumping between the Upper/Lower
# # subplots as it migrates sides - needs BOTH surfaces plotted:
# print(40*'-')
# print('Plotting -Cp surface scatter with stagnation line, average over all frames')
# sv_pressure.plot_variable_surface(
#    lambda s: -sv_pressure.cp(surface=s, stat='mean'),
#    cbar_label='-Cp', span_min=span_min, show_stagnation_line=True,
#    figsize=blade_figsize,
#    savepath=os.path.join(master_path, f'images/cp/cp_surface_with_stagnation_{case}.png'),
# )

# # ------------- Cases comparison plotted into a commo grid ------------- #
# #
# # Cp resampled onto a common (r/R, x/c) grid, compared against a second
# # case with the same geometry (c_ref must be passed explicitly - see
# # README.md for why):
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

# # ------------- Computation with instatenous files containing pressure ------------- #
# 
# # Pressure fluctuation p'(frame) = p(frame) - p_mean, one blade contour
# # per frame - needs a multi-frame file to show a real signal (a
# # single-frame file gives exactly 0 everywhere, since p(frame) == p_mean):
# for frame in range(0, sv_pressure_inst.n_frames, frame_loop_step):
#   print(40*'-')
#   print('Plotting pressure fluctuation, instantaneous frame ', frame)
#   sv_pressure_inst.plot_pressure_fluctuation(
#       frame, span_min=span_min, surface='Upper',
#       figsize=blade_figsize,
#       savepath=os.path.join(master_path, f'images/pfluct/p_fluct_upper_frame{frame:03d}_{case}.png'),
#    )

# # Prms needs no new method - it's already variable(stat='rms'):
# print(40*'-')
# print('Plotting pressure RMS surface scatter, average over all frames')
# sv_pressure_inst.plot_variable_surface(
#    lambda s: sv_pressure_inst.variable('static_pressure', surface=s, stat='rms'),
#    cbar_label='$P_{rms}$ [Pa]', span_min=span_min, surface='Upper',
#    figsize=blade_figsize,
#    savepath=os.path.join(master_path, f'images/pfluct/p_rms_surface_upper_{case}.png'),
# )

# Point time trace + Welch periodogram (wall pressure fluctuations at one
# location, given as % of r/R and x/c - see README.md, "Point time trace").
# Needs a real time axis: pass dt explicitly if this file has no usable
# Metadata/mid_s (see README.md for when that's populated):

# span_pcts = np.arange(86, 101, 2)  # % of r/R
# chord_pcts = np.arange(0, 101, 10)  # % of x/c

# for span in span_pcts:
#   for chord in chord_pcts:
#     print(40*'-')
#     print(f'Plotting pressure time trace at span {span}% and chord {chord}%')
#     sv_pressure_inst.plot_timetrace(
#       'static_pressure', span_pct=span, chord_pct=chord, surface='Upper',
#       ylabel='Static pressure [Pa]', dt=dt,
#       savepath=os.path.join(master_path, f'images/pfluct/spectra/p_timetrace_s{span:03d}_c{chord:03d}_{case}.png'),
#     )
#     print(40*'-')
#     print(f'Plotting pressure periodogram at span {span}% and chord {chord}%')
#     sv_pressure_inst.plot_periodogram(
#       'static_pressure', span_pct=span, chord_pct=chord, surface='Upper', dt=dt,
#       savepath=os.path.join(master_path, f'images/pfluct/spectra/p_periodogram_s{span:03d}_c{chord:03d}_{case}.png'),
#     )
#     print(40*'-')
#     print(f'Exporting pressure time trace at span {span}% and chord {chord}%')
#     sv_pressure_inst.export_timetrace(
#       'static_pressure', span_pct=span, chord_pct=chord, surface='Upper', dt=dt,
#       savepath=os.path.join(master_path, f'data/pfluct/p_timetrace_s{span:03d}_c{chord:03d}_{case}.h5'),
#     )

# # ------------- Strip forces (Hanson's method input) ------------- #

# # Per-radial-strip, time-resolved axial/radial/tangential force, computed
# # directly from a SNCReader.to_h5() file - replaces the manual PowerVIZ
# # "Force Graph" CSV export (ForcesCSVConverter above). span_min isolates
# # one blade (see README.md, "Strip forces" - same reason as everywhere
# # else in this project). Check flip_axial/flip_tangential against what
# # you expect physically before trusting the sign.

# print(40*'-')
# print('Opening StripForces file: ', os.path.join(master_path, inst_force_file))
# # span_min=span_min at load time (see FrictionLines' fl = ... above for why) -
# # every compute()/total_loads() call below already passes span_min=span_min
# # anyway to isolate one blade, so this crops the force field actually
# # read off disk to the same subset, instead of loading the whole rotor.
# sf_avg = StripForces(
#    os.path.join(master_path, avg_force_file),
#    r_tip=r_tip,
#    span_min=span_min,
#    span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis,
#    validate_axes=validate_axes,
# )

# print(40*'-')
# print('Computing strip forces')
# result = sf_avg.compute(span_min=span_min, n_span_bins=10)
# #sf_avg.save(result, os.path.join(master_path, 'data/forces/strip_forces.h5'), dt=dt)

# print(40*'-')
# print('Plotting strip forces bar chart averageg over all frames')
# sf_avg.plot_bar_forces(
#    result, show_totals=False, rho=rho_ref, n_rot=rpm / 60, diameter=2 * r_tip,
#    savepath=os.path.join(master_path, f'images/forces/hanson/strip_forces_bar_avg_{case}.png'),
# )

# # Chordwise-subdivided (non-compact-chord case - see README.md):
# # result_2d = sf.compute(span_min=span_min, n_span_bins=20, n_chord_bins=5)
# # sf.save(result_2d, os.path.join(master_path, 'data/forces/strip_forces_2d.h5'), dt=dt)

# # Integrated totals (thrust/torque/radial/tangential force, independent of
# # strip binning - see README.md, "Integrated totals"). result['totals']
# # is guaranteed consistent with the span_min/span_max compute() above
# # used; total_loads() is the same thing as a standalone call:
# print(40*'-')
# print('thrust [N]:', result['totals']['thrust'].mean())
# print(40*'-')
# print('torque [N.m]:', result['totals']['torque'].mean())
# #totals = sf.total_loads(span_min=span_min)  # standalone, no strip binning needed

# # Thrust/torque coefficients (propeller convention, C_F = F/(rho*n_rot^2*D^4),
# # C_Q = Q/(rho*n_rot^2*D^5) - see README.md, "Thrust/torque coefficients").
# # n_rot is rev/s, NOT RPM:
# print(40*'-')
# print('Plotting strip forces bar chart averageg over all frames with non-dimensional coefficients')
# sf_avg.plot_bar_forces(
#    result, show_totals=False, rho=rho_ref, n_rot=rpm / 60, diameter=2 * r_tip,
#    savepath=os.path.join(master_path, f'images/forces/hanson/strip_forces_bar_coeffs_avg_{case}.png'),
# )

# # Physical radius instead of r/R on the x-axis:
# sf.plot_bar_forces(
#    result, show_totals=True, normalize_radius=False,
#    savepath=os.path.join(master_path, 'images/forces/strip_forces_bar_radius.png'),
# )

# ------------- Time domain / phase-locked / harmonics (Hanson's method) ------------- #
#
# Only meaningful on an "inst" (multi-frame/transient) file - see
# README.md, "Average vs. instantaneous cases". Needs rpm (set on
# StripForces itself, not compute()) for phase_lock()/harmonics().

print(40*'-')
print('Opening StripForces file: ', os.path.join(master_path, inst_force_file))
# span_min=span_min at load time - see sf_avg above. This is the big
# multi-frame/transient file, so this is the crop that actually matters
# for memory (the one that OOM-killed a real ~660 GB whole-rotor case
# before this parameter existed).
sf_inst = StripForces(
   os.path.join(master_path, inst_force_file),
   r_tip=r_tip, rpm=rpm,
   span_min=span_min,
   span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis,
   validate_axes=validate_axes,
)

print(40*'-')
print('Computing instantaneous strip forces')
result_inst = sf_inst.compute(span_min=span_min, n_span_bins=10)

# Raw per-strip time trace (see README.md, "Time trace"):
print(40*'-')
print('Plotting instantaneous strip forces time trace for the axial component')
sf_inst.plot_time_trace(
   result_inst, dt=dt, component='axial', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_time_trace_axial_{case}.png'),
)

print(40*'-')
print('Plotting instantaneous strip forces time trace for the radial component')
sf_inst.plot_time_trace(
   result_inst, dt=dt, component='radial', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_time_trace_radial_{case}.png'),
)

print(40*'-')
print('Plotting instantaneous strip forces time trace for the tangential component')
sf_inst.plot_time_trace(
   result_inst, dt=dt, component='tangential', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_time_trace_tangential_{case}.png'),
)

# Phase-locked (revolution-folded) force vs azimuth (see README.md,
# "Phase-locked (revolution-folded) forces"):
print(40*'-')
print('Plotting phase-locked forces vs azimuth for the axial component')
phase_locked = sf_inst.phase_lock(result_inst, dt=dt, n_azimuth_bins=72)
sf_inst.plot_vs_angle(
   phase_locked, component='axial', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_vs_angle_axial_{case}.png'),
)

print(40*'-')
print('Plotting phase-locked forces vs azimuth for the radial component')
sf_inst.plot_vs_angle(
   phase_locked, component='radial', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_vs_angle_radial_{case}.png'),
)

print(40*'-')
print('Plotting phase-locked forces vs azimuth for the tangential component')
sf_inst.plot_vs_angle(
   phase_locked, component='tangential', strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_vs_angle_tangential_{case}.png'),
)

# Harmonics of the rotation frequency - Hanson's method's actual |F_n(r)|
# input (see README.md, "Harmonics (Hanson's method's actual input)"):
print(40*'-')
print('Plotting harmonics for the axial component')
h = sf_inst.harmonics(result_inst, dt=dt, component='axial', n_harmonics=17)
sf_inst.plot_harmonics(
   h, strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonics_axial_{case}.png'),
)

print(40*'-')
print('Plotting harmonics for the radial component')
h = sf_inst.harmonics(result_inst, dt=dt, component='radial', n_harmonics=17)
sf_inst.plot_harmonics(
   h, strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonics_radial_{case}.png'),
)

print(40*'-')
print('Plotting harmonics for the tangential component')
h = sf_inst.harmonics(result_inst, dt=dt, component='tangential', n_harmonics=17)
sf_inst.plot_harmonics(
   h, strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonics_tangential_{case}.png'),
)

# With phase (needed before actually handing this to Hanson's model, or
# to check a harmonic's peak azimuth against a known physical cause -
# see README.md, "Phase"):
print(40*'-')
print('Plotting harmonics with phase for the axial component')
h_phase = sf_inst.harmonics(result_inst, dt=dt, component='axial', n_harmonics=17, return_phase=True)
sf_inst.plot_harmonics(
   h_phase, strips=[0, 2, 4, 6, 8, 9], show_phase=True,
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonics_phase_axial_{case}.png'),
)

print(40*'-')
print('Computing peak azimuth of each harmonic for the axial component')
peak_deg = sf_inst.peak_azimuth(h_phase)  # (n_harmonics, n_span_bins)
print(40*'-')
print('Axial component peak azimuth of each harmonic (deg): ', peak_deg)

print(40*'-')
print('Plotting harmonics with phase for the radial component')
h_phase = sf_inst.harmonics(result_inst, dt=dt, component='radial', n_harmonics=17, return_phase=True)
sf_inst.plot_harmonics(
   h_phase, strips=[0, 2, 4, 6, 8, 9], show_phase=True,
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonics_phase_radial_{case}.png'),
)

print(40*'-')
print('Computing peak azimuth of each harmonic for the radial component')
peak_deg = sf_inst.peak_azimuth(h_phase)  # (n_harmonics, n_span_bins)
print(40*'-')
print('Radial component peak azimuth of each harmonic (deg): ', peak_deg)


print(40*'-')
print('Plotting harmonics with phase for the tangential component')
h_phase = sf_inst.harmonics(result_inst, dt=dt, component='tangential', n_harmonics=17, return_phase=True)
sf_inst.plot_harmonics(
   h_phase, strips=[0, 2, 4, 6, 8, 9], show_phase=True,
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonics_phase_tangential_{case}.png'),
)

print(40*'-')
print('Computing peak azimuth of each harmonic for the tangential component')
peak_deg = sf_inst.peak_azimuth(h_phase)  # (n_harmonics, n_span_bins)
print(40*'-')
print('Tangential component peak azimuth of each harmonic (deg): ', peak_deg)


# Reconstruction check against phase_lock()'s own empirical curve:
#phase_locked = sf_inst.phase_lock(result_inst, dt=dt, n_azimuth_bins=72)
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
totals_inst = sf_inst.total_loads(span_min=span_min)  # standalone total, same span as result_inst above
sf_inst.plot_phase_portrait(
   totals_inst, component_pair=('axial', 'tangential'),
   savepath=os.path.join(master_path, f'images/forces/convergence/phase_portrait_axial_tangential_{case}.png'),
)

print(40*'-')
print('Plotting whole-blade phase portrait (axial vs radial)')
totals_inst = sf_inst.total_loads(span_min=span_min)  # standalone total, same span as result_inst above
sf_inst.plot_phase_portrait(
   totals_inst, component_pair=('axial', 'radial'),
   savepath=os.path.join(master_path, f'images/forces/convergence/phase_portrait_axial_radial_{case}.png'),
)

print(40*'-')
print('Plotting whole-blade phase portrait (radial vs tangential)')
totals_inst = sf_inst.total_loads(span_min=span_min)  # standalone total, same span as result_inst above
sf_inst.plot_phase_portrait(
   totals_inst, component_pair=('tangential', 'radial'),
   savepath=os.path.join(master_path, f'images/forces/convergence/phase_portrait_tangential_radial_{case}.png'),
)


print(40*'-')
print('Plotting per-strip phase portraits (axial vs radial) - localizes convergence issues by span')
sf_inst.plot_phase_portrait_by_strip(
   result_inst, component_pair=('axial', 'radial'), strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/convergence/phase_portrait_by_strip_axial_radial_{case}.png'),
)

print(40*'-')
print('Plotting per-strip phase portraits (axial vs tangential) - localizes convergence issues by span')
sf_inst.plot_phase_portrait_by_strip(
   result_inst, component_pair=('axial', 'tangential'), strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/convergence/phase_portrait_by_strip_axial_tangential_{case}.png'),
)

print(40*'-')
print('Plotting per-strip phase portraits (tangential vs radial) - localizes convergence issues by span')
sf_inst.plot_phase_portrait_by_strip(
   result_inst, component_pair=('tangential', 'radial'), strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/convergence/phase_portrait_by_strip_tangential_radial_{case}.png'),
)

# Convergence checking: cumulative (running) mean vs revolutions included
# (see README.md, "Convergence checking: running/cumulative mean") - a
# converged quantity's running mean flattens to a horizontal asymptote.
# Not tied to StripForces specifically - takes any plain 1D array:
print(40*'-')
print('Plotting cumulative mean of thrust vs revolutions included')
plot_cumulative_mean(
   totals_inst['thrust'], dt=dt, rpm=rpm, ylabel='Thrust [N]',
   savepath=os.path.join(master_path, f'images/forces/convergence/thrust_cumulative_mean_{case}.png'),
)

print(40*'-')
print('Plotting cumulative mean of thrust vs revolutions included')
plot_cumulative_mean(
   totals_inst['torque'], dt=dt, rpm=rpm, ylabel='Torque [Nm]',
   savepath=os.path.join(master_path, f'images/forces/convergence/torque_cumulative_mean_{case}.png'),
)

# Mean AND variance together (Pope's <U>/<u'^2> pair). sync='none'
# (every frame - fine for an isolated rotor in hover, where each frame
# is already a reasonably independent-ish realization); use
# sync='revolution' (needs dt+rpm) INSTEAD if the signal has a real
# once-per-revolution component - REQUIRED then to see a clean
# asymptote (see README.md, "Mean AND variance together, synced to
# revolution boundaries" - a plain per-frame running mean of such a
# signal shows a persistent ripple that this removes); or
# sync='periodicity' (needs dt+rpm+period_deg) for a case whose real
# periodicity is SHORTER than one revolution (e.g. a 4-blade rotor /
# 4-vane stator interaction repeating every 360/4=90 degrees):
print(40*'-')
print('Plotting cumulative mean+variance of thrust, synced to revolution boundaries')
plot_cumulative_stats(
   totals_inst['thrust'], dt=dt, rpm=rpm, sync='none', ylabel='Thrust [N]',
   savepath=os.path.join(master_path, f'images/forces/convergence/thrust_cumulative_stats_{case}.png'),
)

print(40*'-')
print('Plotting cumulative mean+variance of thrust, synced to revolution boundaries')
plot_cumulative_stats(
   totals_inst['torque'], dt=dt, rpm=rpm, sync='none', ylabel='Torque [Nm]',
   savepath=os.path.join(master_path, f'images/forces/convergence/torque_cumulative_stats_{case}.png'),
)

# Example for a rotor-stator case instead (NOT this project's isolated
# rotor - shown for reference): 4 blades / 4 vanes repeat every
# 360/4=90 degrees, so sync every 90 degrees rather than every full
# revolution to get 4x the comparable-phase samples per run:
# plot_cumulative_stats(
#   totals_inst['thrust'], dt=dt, rpm=rpm, sync='periodicity', period_deg=90.0,
#   ylabel='Thrust [N]',
#   savepath=os.path.join(master_path, f'images/forces/convergence/thrust_cumulative_stats_periodicity_{case}.png'),
# )

# Convergence checking: 3rd/4th-order statistics (running skewness and
# flatness - see README.md, "Convergence checking: higher-order
# moments (skewness/flatness)"). Needs substantially more revolutions
# to converge than the mean/variance above - don't expect it to flatten
# as quickly:
print(40*'-')
print('Plotting cumulative skewness+flatness of thrust')
plot_cumulative_moments(
   totals_inst['thrust'], dt=dt, rpm=rpm, sync='none', label='Thrust',
   savepath=os.path.join(master_path, f'images/forces/convergence/thrust_cumulative_moments_{case}.png'),
)

print(40*'-')
print('Plotting cumulative skewness+flatness of torque')
plot_cumulative_moments(
   totals_inst['torque'], dt=dt, rpm=rpm, sync='none', label='Torque',
   savepath=os.path.join(master_path, f'images/forces/convergence/torque_cumulative_moments_{case}.png'),
)

# Convergence checking: autocorrelation comparison between independent
# windows (see README.md, "Convergence checking: autocorrelation" - NOT
# a single-window "is rho(s) even" check, which is guaranteed to pass
# trivially regardless of convergence - comparing INDEPENDENT windows is
# what's actually meaningful):

print(40*'-')
print('Plotting thrust autocorrelation, first half vs second half of the run')
plot_autocorrelation_windows(
   totals_inst['thrust'], n_windows=2, dt=dt, labels=['First half', 'Second half'],
   savepath=os.path.join(master_path, f'images/forces/convergence/thrust_autocorrelation_windows_{case}.png'),
)

print(40*'-')
print('Plotting torque autocorrelation, first half vs second half of the run')
plot_autocorrelation_windows(
   totals_inst['torque'], n_windows=2, dt=dt, labels=['First half', 'Second half'],
   savepath=os.path.join(master_path, f'images/forces/convergence/torque_autocorrelation_windows_{case}.png'),
)

# Convergence checking: statistical uncertainty of the mean (SEM), from
# the signal's own integral timescale (see README.md, "Convergence
# checking: statistical uncertainty of the mean") - reuses the same
# autocorrelation machinery above, but turns it into an actual error
# bar on thrust/torque instead of just an eyeballed plot. sync='none'
# here for the same reason as cumulative_stats above (isolated rotor in
# hover); switch to 'revolution'/'periodicity' for a case with a real
# periodic component - see standard_error()'s docstring for why (the
# raw per-frame autocorrelation of a periodic signal never decays to
# zero, which corrupts the integral-timescale estimate).
print(40*'-')
print('Estimating standard error of the mean thrust')
stats = standard_error(totals_inst['thrust'], dt=dt, rpm=rpm, sync='none')
print(f"mean={stats['mean']:.4g} N, sigma={stats['sigma']:.4g} N, T_int={stats['T_int']:.4g} s, "
      f"n_eff={stats['n_eff']:.1f}, SEM={stats['sem']:.4g} N ({stats['relative_sem']*100:.3f}% of mean)")

# Same thing, plotted - the rho(s) curve with the actually-integrated
# region shaded and T_int/SEM/n_eff/revolutions_required as an inset
# (same style as plot_bar_forces()'s show_totals) - target_relative_sem
# defaults to 0.01 (1% of the mean), override for a stricter/looser target:
plot_integral_timescale(
   totals_inst['thrust'], dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
   savepath=os.path.join(master_path, f'images/forces/convergence/thrust_integral_timescale_{case}.png'),
)

plot_integral_timescale(
   totals_inst['torque'], dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
   savepath=os.path.join(master_path, f'images/forces/convergence/torque_integral_timescale_{case}.png'),
)

# How many MORE revolutions to reach a target precision (e.g. 0.1%
# relative SEM on thrust) - answers "how much longer do I need to run
# this" with an actual number instead of a guess:
req = required_averaging_time(
   totals_inst['thrust'], dt=dt, rpm=rpm, sync='none', target_relative_sem=0.001,
)
print(f"Need {req['revolutions_required']:.1f} total revolutions for 0.1% relative SEM "
      f"({req['revolutions_current']:.1f} already run, "
      f"{req['additional_revolutions']:.1f} more needed)")

# Convergence checking: cycle-to-cycle waveform correlation (see
# README.md, "Convergence checking: cycle-to-cycle correlation") -
# checks a DIFFERENT thing than everything above: not whether a running
# STATISTIC has flattened, but whether the per-revolution WAVEFORM
# SHAPE has stopped changing - the actual assumption phase_lock()/
# harmonics() below and TipVortexPhaseAverage depend on. period_deg=360
# (default) = one full revolution:
# print(40*'-')
# print('Plotting cycle-to-cycle correlation of thrust')
# plot_cycle_correlation(
#    totals_inst['thrust'], dt=dt, rpm=rpm, period_deg=360.0,
#    savepath=os.path.join(master_path, f'images/forces/convergence/thrust_cycle_correlation_{case}.png'),
# )

# # Rotor-stator example instead (NOT this project's isolated rotor -
# # shown for reference): correlate every 90-degree interaction period
# # rather than every full revolution:
# plot_cycle_correlation(
#   totals_inst['thrust'], dt=dt, rpm=rpm,,period_deg=90.0,
#   savepath=os.path.join(master_path, f'images/forces/convergence/thrust_cycle_correlation_90deg_{case}.png'),
# )

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
