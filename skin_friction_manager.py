import os
import numpy as np

from case_config import load_case_config
from bladeprocessor.friction_lines import FrictionLines
from bladeprocessor.surface_variable import SurfaceVariable
from bladeprocessor.convergence import plot_cumulative_stats
from bladeprocessor.convergence import plot_cumulative_moments
from bladeprocessor.convergence import plot_autocorrelation_windows
from bladeprocessor.convergence import plot_integral_timescale

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
# (q_ref = 0.5*rho_ref*(omega*r)^2 - see README.md's "Equations" section
# for the full derivation).

cfg = load_case_config()
master_path = cfg.master_path
case = cfg.case
inst_force_file = cfg.inst_force_file
avg_force_file = cfg.avg_force_file
r_tip = cfg.r_tip
rho_ref = cfg.rho_ref
rpm = cfg.rpm
c_ref = cfg.c_ref
normalize = cfg.normalize
span_axis = cfg.span_axis
chord_axis = cfg.chord_axis
thickness_axis = cfg.thickness_axis
validate_axes = cfg.validate_axes
span_min = cfg.span_min
reverse_chord = cfg.reverse_chord
radii = cfg.radii
dt = cfg.dt
blade_figsize = cfg.blade_figsize
frame_loop_step = cfg.frame_loop_step

for _sub in ('cf/avg', 'cf/inst', 'cf/rms', 'cf/convergence/global', 'cf/convergence/local'):
	os.makedirs(os.path.join(master_path, 'images', _sub), exist_ok=True)


# ------------- Friction related post-processing ------------- #
print(40*'-')
print('Opening FrictionLines file: ', os.path.join(master_path, inst_force_file))
# span_min=span_min here (not just on every call below) - for a whole-rotor
# case with no separate blade parts to select via face_name at
# conversion time, every single call below already passes span_min=span_min
# anyway (to isolate one blade - see the note further down), so cropping
# at load time means the (potentially huge) force field is only ever
# read for the surviving ~half of the points, not the whole rotor - see
# FrictionLines.__init__'s span_min/span_max docstring. Fixed a real OOM
# on a ~660 GB case this way.
fl = FrictionLines(
   os.path.join(master_path, inst_force_file),
   r_tip=r_tip,
   rho_ref=rho_ref,
   rpm=rpm,
   c_ref=c_ref,
   span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis,
   span_min=span_min, validate_axes=validate_axes
)

# Dimensional wall shear vector (tau = F - (F.n)n), no rho_ref/rpm needed:
# tau = fl.wall_shear(surface='Upper', frame=None)  # frame=None -> average over every frame in the file

# Cf magnitude and signed chordwise/spanwise components, one frame or the average:
cf_mag = fl.cf(surface='Upper', frame=None, component=None)
print(40*'-')
print('Average Cf magnitude: ', np.mean(cf_mag))
# cf_chordwise_frame0 = fl.cf(surface='Upper', frame=0, component='chordwise')

for frame in range(0,fl.n_frames,frame_loop_step):
	cf_mag = fl.cf(surface='Upper', frame=frame, component=None)
	print(40*'-')
	print(f'Cf magnitude: {np.mean(cf_mag)} at frame {frame:03d}')

# Cf vs local x/c at several radii, one plot per call - instantaneous and
# average. span_min isolates one blade (REQUIRED in practice - without it,
# a radius band mixes both blades' chord ranges and produces a spurious
# double peak, see README.md); reverse_chord fixes which end is the
# leading vs. trailing edge (no automatic detection - check per case, Cf
# should peak sharply near x/c=0 and decay toward x/c=1; if it's flipped,
# set reverse_chord=reverse_chord - see README.md's "Two bugs found and fixed"):

print(40*'-')
print('Plotting Cf vs x/c at several radii, Upper surface, average over all frames')
print(40*'-')
print('Plotting Cf vs x/c magnitude')
fl.plot_cf_radii(
   radii=radii,#[0.045, 0.072, 0.100, 0.117, 0.122],
   frame=None, component=None, span_min=span_min, reverse_chord=reverse_chord,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/avg/cf_radii_mag_avg_{case}.png'),
)
print(40*'-')
print('Plotting Cf vs x/c chordwise component')
fl.plot_cf_radii(
   radii=radii,
   frame=None, component='chordwise', span_min=span_min, reverse_chord=reverse_chord,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/avg/cf_radii_chordwise_avg_{case}.png'),
)
print(40*'-')
print('Plotting Cf vs x/c spanwise component')
fl.plot_cf_radii(
   radii=radii,
   frame=None, component='spanwise', span_min=span_min, reverse_chord=reverse_chord,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/avg/cf_radii_spanwise_avg_{case}.png'),
)

for frame in range(0,fl.n_frames,frame_loop_step):
	print(40*'-')
	print(f'Plotting Cf vs x/c at several radii, Upper surface, average for frame {frame:03d}')
	print(40*'-')
	print('Plotting Cf vs x/c magnitude')
	fl.plot_cf_radii(
	radii=radii,
	frame=frame, component=None, span_min=span_min, reverse_chord=reverse_chord,
	normalize=normalize,
	savepath=os.path.join(master_path, f'images/cf/inst/cf_radii_mag_frame{frame:03d}_{case}.png'),
	)
	print(40*'-')
	print('Plotting Cf vs x/c chordwise component')
	fl.plot_cf_radii(
	radii=radii,
	frame=frame, component='chordwise', span_min=span_min, reverse_chord=reverse_chord,
	normalize=normalize,
	savepath=os.path.join(master_path, f'images/cf/inst/cf_radii_chordwise_frame{frame:03d}_{case}.png'),
	)
	print(40*'-')
	print('Plotting Cf vs x/c spanwise component')
	fl.plot_cf_radii(
	radii=radii,
	frame=frame, component='spanwise', span_min=span_min, reverse_chord=reverse_chord,
	normalize=normalize,
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
   radii=radii,
   surface='Upper', frame=None, stat='rms', span_min=span_min, reverse_chord=reverse_chord,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/rms/cf_rms_radii_avg_{case}.png'),
)
print(40*'-')
print('Plotting Cf RMS vs x/c color map')
fl.friction_lines(
   surface='Upper', frame=None, stat='rms', span_min=span_min,
   figsize=blade_figsize,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/rms/cf_rms_map_{case}.png'),
)

# Friction lines (Upper+Lower stacked by default) - span_min isolates one
# blade half on a two-bladed rotor centered at span=0 (see the method's
# docstring - there's no reliable automatic hub cutoff, pass what's right
# for this case's mesh):
print(40*'-')
print('Plotting Friction Lines, Upper surface, average over all frames')
fl.friction_lines(
   frame=None, span_min=span_min, surface='Upper',
   figsize=blade_figsize,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/avg/friction_lines_avg_{case}.png'),
)

for frame in range(0,fl.n_frames,frame_loop_step):
	print(40*'-')
	print(f'Plotting Friction Lines, Upper surface, for frame {frame:03d}')
	fl.friction_lines(
	   frame=frame, span_min=span_min, surface='Upper',
	   figsize=blade_figsize,
	   normalize=normalize,
	   savepath=os.path.join(master_path, f'images/cf/inst/friction_lines_frame{frame:03d}_{case}.png'),
	)

# Separation/reattachment line (chordwise-Cf sign crossings) - restricted
# to one blade section via span_min/span_max like everything else here;
# reverse_chord must match what plot_cf_radii()/cf_at_radii() needed on
# this case (see README.md, "Separation/reattachment line"):
#sep_points = fl.separation_line(surface='Upper', frame=None, span_min=span_min, reverse_chord=reverse_chord)
#fl.save_separation_line(sep_points, os.path.join(master_path, 'data/cf/separation_line.txt'))

# Overlaid directly on friction_lines() (separation in red, reattachment in cyan):
print(40*'-')
print('Plotting Friction Lines with separation/reattachment line, Upper surface, average over all frames')
fl.friction_lines(
   surface='Upper', frame=None, span_min=span_min, show_separation_line=True,
   separation_line_kwargs={'reverse_chord': reverse_chord},
   figsize=blade_figsize,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/avg/friction_lines_separation_{case}.png'),
)

for frame in range(0,fl.n_frames,frame_loop_step):
	print(40*'-')
	print(f'Plotting Friction Lines with separation/reattachment line, Upper surface, for frame {frame:03d}')
	fl.friction_lines(
	   surface='Upper', frame=frame, span_min=span_min, show_separation_line=True,
	   separation_line_kwargs={'reverse_chord': reverse_chord},
	   figsize=blade_figsize,
	   normalize=normalize,
	   savepath=os.path.join(master_path, f'images/cf/inst/friction_lines_separation_frame{frame:03d}_{case}.png'),
	)

# # Spanwise migration-reversal line (spanwise-Cf sign crossings - a
# # DIFFERENT physical phenomenon from separation/reattachment above, see
# # README.md, "Spanwise migration-reversal line"). edge_crop (default
# # 0.05) excludes crossings too close to the LE/TE - real LE noise on
# # this case, found and fixed this way after two amplitude-based filter
# # attempts backfired (see the README section and migration_line()'s own
# # docstring for the full story):
# #mig_points = fl.migration_line(surface='Upper', frame=None, span_min=span_min, reverse_chord=reverse_chord)
# #fl.save_migration_line(mig_points, os.path.join(master_path, 'data/cf/migration_line.txt'))

#fl.friction_lines(
#    surface='Upper', frame=None, span_min=span_min, show_migration_line=True,
#    migration_line_kwargs={'reverse_chord': True},
#    figsize=blade_figsize,
#    savepath=os.path.join(master_path, 'images/cf/friction_lines_migration.png'),
#)

# Vortex-footprint critical points (node/saddle/focus - see README.md,
# "Vortex-footprint critical points"; 'focus' = actual vortex core, e.g.
# a leading-edge or corner/horseshoe vortex, not just an ordinary
# separation/reattachment feature). No reverse_chord - works in raw
# physical (span, chord) coordinates, not x/c:
#crit_points = fl.critical_points(surface='Upper', frame=None, span_min=span_min)
#fl.save_critical_points(crit_points, '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/cf/critical_points.txt')
#print('Poincare index N+F-S =', fl.poincare_index(crit_points))  # see README.md - NOT expected to be 2 on this open, cropped selection

# show_critical_points_index=True annotates the figure itself with N+F-S:
print(40*'-')
print('Plotting Friction Lines with critical points, Upper surface, average over all frames')
fl.friction_lines(
   surface='Upper', frame=None, span_min=span_min, show_critical_points=True, show_critical_points_index=False,
   figsize=blade_figsize,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/avg/friction_lines_critical_points_{case}.png'),
)

for frame in range(0,fl.n_frames,frame_loop_step):
	print(40*'-')
	print(f'Plotting Friction Lines with critical points, Upper surface, for frame {frame:03d}')
	fl.friction_lines(
	surface='Upper', frame=frame, span_min=span_min, show_critical_points=True, show_critical_points_index=False,
	figsize=blade_figsize,
	normalize=normalize,
	savepath=os.path.join(master_path, f'images/cf/inst/friction_lines_critical_points_frame{frame:03d}_{case}.png'),
	)

# ------------- Any surface variable at radii/over the blade (Cf-related) ------------- #
#
# Skin Friction as a SurfaceVariable (generalizes friction_lines() above
# to any scalar field - see README.md, "Any surface variable at radii" /
# "Whole-blade surface plot"). Input: a SNCReader.to_h5(..., surface_split=True)
# file (the forces branch, same as fl above), for the raw Skin_Friction
# variable rather than the tau-derived Cf FrictionLines computes:

print(40*'-')
print('Opening SurfaceVariable file: ', os.path.join(master_path, inst_force_file))
sv_inst_forces = SurfaceVariable(
   os.path.join(master_path, inst_force_file),
   r_tip=r_tip,
   rho_ref=rho_ref,
   rpm=rpm,
   c_ref=c_ref,
   span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis
)

print(40*'-')
print('Plotting Skin Friction surface scatter, average over all frames')
sv_inst_forces.plot_variable_surface(
   lambda s: sv_inst_forces.variable('Skin_Friction', surface=s, stat='mean'),
   cbar_label='Skin Friction [Pa]', span_min=span_min, surface='Upper',
   figsize=blade_figsize,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/avg/cf_surface_avg_upper_{case}.png'),
)

print(40*'-')
print('Plotting Skin Friction surface scatter, average over all frames')
sv_inst_forces.plot_variable_surface(
   lambda s: sv_inst_forces.variable('Skin_Friction', surface=s, stat='rms'),
   cbar_label='Skin Friction [Pa]', span_min=span_min, surface='Upper',
   figsize=blade_figsize,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/rms/cf_surface_rms_upper_{case}.png'),
)

for frame in range(0,sv_inst_forces.n_frames,frame_loop_step):
   
   print(40*'-')
   print('Plotting Skin Friction surface scatter, average over all frames')
   sv_inst_forces.plot_variable_surface(
      lambda s: sv_inst_forces.variable('Skin_Friction', surface=s, frame=frame),
      cbar_label='Skin Friction [Pa]', span_min=span_min, surface='Upper',
      figsize=blade_figsize,
      normalize=normalize,
      savepath=os.path.join(master_path, f'images/cf/inst/cf_surface_inst_upper_frame{frame:03d}_{case}.png'),
   )

# Convergence checking: Cf phase portrait (see README.md, "Convergence
# checking: wall-shear/Cf phase portraits") - near-wall/viscous
# quantities converge MORE SLOWLY than integrated forces, so this needs
# checking separately from StripForces' phase portraits even if those
# already look converged:
print(40*'-')
print('Plotting Cf phase portrait (magnitude vs chordwise), Upper surface')
fl.plot_cf_phase_portrait(
   component_pair=(None, 'chordwise'), surface='Upper', span_min=span_min,
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_phase_portrait_mag_chordwise_{case}.png'),
)

print(40*'-')
print('Plotting Cf phase portrait (magnitude vs spanwise), Upper surface')
fl.plot_cf_phase_portrait(
   component_pair=(None, 'spanwise'), surface='Upper', span_min=span_min,
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_phase_portrait_mag_spanwise_{case}.png'),
)

print(40*'-')
print('Plotting Cf phase portrait (magnitude vs spanwise), Upper surface')
fl.plot_cf_phase_portrait(
   component_pair=('spanwise', 'chordwise'), surface='Upper', span_min=span_min,
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_phase_portrait_spanwise_chordwise_{case}.png'),
)

print(40*'-')
print('Plotting per-strip Cf phase portraits (spanwise vs chordwise) - localizes convergence issues by span')
fl.plot_cf_phase_portrait_by_strip(
   component_pair=(None, 'chordwise'), surface='Upper', span_min=span_min, n_span_bins=10, strips=[2, 4, 6, 8, 9],
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_phase_portrait_mag_chordwise_by_strip_{case}.png'),
)

print(40*'-')
print('Plotting per-strip Cf phase portraits (spanwise vs chordwise) - localizes convergence issues by span')
fl.plot_cf_phase_portrait_by_strip(
   component_pair=(None, 'spanwise'), surface='Upper', span_min=span_min, n_span_bins=10, strips=[2, 4, 6, 8, 9],
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_phase_portrait_mag_spanwise_by_strip_{case}.png'),
)

print(40*'-')
print('Plotting per-strip Cf phase portraits (spanwise vs chordwise) - localizes convergence issues by span')
fl.plot_cf_phase_portrait_by_strip(
   component_pair=('spanwise', 'chordwise'), surface='Upper', span_min=span_min, n_span_bins=10, strips=[2, 4, 6, 8, 9],
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_phase_portrait_spanwise_chordwise_by_strip_{case}.png'),
)

# Convergence checking: cumulative mean+variance and higher-order
# moments (skewness/flatness) of Cf ITSELF - the same tools already
# used for thrust/torque in forces_manager.py, fed with Cf's own
# spatial-mean-per-frame series instead. Near-wall/viscous quantities
# are known to converge MORE SLOWLY than integrated forces (see the Cf
# phase-portrait note above), so this is worth checking even once
# thrust/torque already look converged. No extra span cropping needed
# here - fl was already constructed with span_min=span_min, so
# cf_time_series() only ever covers the one blade half fl was built
# with; just reduce it to one scalar per frame:
cf_mag_series = fl.cf_time_series(surface='Upper', component=None).mean(axis=1)
cf_chordwise_series = fl.cf_time_series(surface='Upper', component='chordwise').mean(axis=1)
cf_spanwise_series = fl.cf_time_series(surface='Upper', component='spanwise').mean(axis=1)

print(40*'-')
print('Plotting cumulative mean+variance of Cf magnitude')
plot_cumulative_stats(
   cf_mag_series, dt=dt, rpm=rpm, sync='none', ylabel='$C_f$ [-]',
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_mag_cumulative_stats_{case}.png'),
)

print(40*'-')
print('Plotting cumulative mean+variance of chordwise Cf')
plot_cumulative_stats(
   cf_chordwise_series, dt=dt, rpm=rpm, sync='none', ylabel='$C_{f,chordwise}$ [-]',
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_chordwise_cumulative_stats_{case}.png'),
)

print(40*'-')
print('Plotting cumulative mean+variance of spanwise Cf')
plot_cumulative_stats(
   cf_spanwise_series, dt=dt, rpm=rpm, sync='none', ylabel='$C_{f,spanwise}$ [-]',
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_spanwise_cumulative_stats_{case}.png'),
)

print(40*'-')
print('Plotting cumulative skewness+flatness of Cf magnitude')
plot_cumulative_moments(
   cf_mag_series, dt=dt, rpm=rpm, sync='none', label='$C_f$',
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_mag_cumulative_moments_{case}.png'),
)

print(40*'-')
print('Plotting cumulative skewness+flatness of chordwise Cf')
plot_cumulative_moments(
   cf_chordwise_series, dt=dt, rpm=rpm, sync='none', label='$C_{f,chordwise}$',
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_chordwise_cumulative_moments_{case}.png'),
)

print(40*'-')
print('Plotting cumulative skewness+flatness of spanwise Cf')
plot_cumulative_moments(
   cf_spanwise_series, dt=dt, rpm=rpm, sync='none', label='$C_{f,spanwise}$',
   savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_spanwise_cumulative_moments_{case}.png'),
)

print(40*'-')
print('Plotting cf magnitude autocorrelation, first half vs second half of the run')
plot_autocorrelation_windows(
  cf_mag_series, n_windows=2, dt=dt, labels=['First half', 'Second half'],
  savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_mag_autocorrelation_windows_{case}.png'),
)

print(40*'-')
print('Plotting cf spanwise autocorrelation, first half vs second half of the run')
plot_autocorrelation_windows(
  cf_spanwise_series, n_windows=2, dt=dt, labels=['First half', 'Second half'],
  savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_spanwise_autocorrelation_windows_{case}.png'),
)

print(40*'-')
print('Plotting cf chordwise autocorrelation, first half vs second half of the run')
plot_autocorrelation_windows(
  cf_chordwise_series, n_windows=2, dt=dt, labels=['First half', 'Second half'],
  savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_chordwise_autocorrelation_windows_{case}.png'),
)

print(40*'-')
print('Plotting integral timescale / required averaging time for cf magnitude')
plot_integral_timescale(
  cf_mag_series, dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
  savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_mag_integral_timescale_{case}.png'),
)

print(40*'-')
print('Plotting integral timescale / required averaging time for cf spanwise')
plot_integral_timescale(
  cf_spanwise_series, dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
  savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_spanwise_integral_timescale_{case}.png'),
)

print(40*'-')
print('Plotting integral timescale / required averaging time for cf chordwise')
plot_integral_timescale(
  cf_chordwise_series, dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
  savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_chordwise_integral_timescale_{case}.png'),
)

# # Rolling (fixed-size sliding window) overlay on top of the cumulative
# # curve above (see README.md, "Cross-checking the cumulative curve with
# # a fixed-size rolling window") - a cumulative statistic's sensitivity
# # to new data shrinks as 1/n, so a late-run drift can hide behind an
# # already-flat-looking cumulative curve; a fixed-length window stays
# # equally sensitive throughout. window has no good universal default -
# # pick something like a few times integral_timescale()'s own T_int
# # (converted to a sample count) once that's been computed for this
# # case, not blindly reused from another one.
# plot_cumulative_stats(
#    cf_mag_series, dt=dt, rpm=rpm, sync='none', window=200, ylabel='$C_f$ [-]',
#    savepath=os.path.join(master_path, f'images/cf/convergence/global/cf_mag_cumulative_stats_rolling_{case}.png'),
# )

# Convergence at a SINGLE point, not just the spatial mean above (see
# README.md, "Convergence checking at a single point, not just the
# spatial mean") - a converged spatial mean is necessary but not
# sufficient for convergence at any given point; span_pct/chord_pct=90/25
# targets a point near the tip, a location expected to be among the
# hardest to converge (see friction_lines_test_migration_overlay.png-
# style separation/reattachment discussion above for why the tip region
# is a reasonable "hardest case" choice on this project's own geometry -
# adjust per case).
chord_pts = np.arange(0, 101, 10)
span_pts = [50, 70, 80, 90]

for chord in chord_pts:
   for span in span_pts:
      # Magnitude
      print(40*'-')
      print('Plotting cumulative mean+variance of Cf magnitude at a single point near the tip')
      cf_mag_point_series, cf_mag_point_info = fl.cf_time_series_at_point(span_pct=span, chord_pct=chord, surface='Upper')

      print('Point actually used: ', cf_mag_point_info)
      plot_cumulative_stats(
         cf_mag_point_series, dt=dt, rpm=rpm, sync='none', ylabel='$C_f$ [-]',
         savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_mag_point_cumulative_stats_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      print(40*'-')
      print('Plotting cumulative skewness+flatness of spanwise Cf')
      plot_cumulative_moments(
         cf_mag_point_series, dt=dt, rpm=rpm, sync='none', label='$C_{f,spanwise}$',
         savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_mag_point_cumulative_moments_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      print(40*'-')
      print('Plotting cf magnitude autocorrelation, first half vs second half of the run')
      plot_autocorrelation_windows(
      cf_mag_point_series, n_windows=2, dt=dt, labels=['First half', 'Second half'],
      savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_mag_point_autocorrelation_windows_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      print(40*'-')
      print('Plotting integral timescale / required averaging time for cf magnitude')
      plot_integral_timescale(
      cf_mag_point_series, dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
      savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_mag_point_integral_timescale_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      # Chordwise
      print(40*'-')
      print('Plotting cumulative mean+variance of Cf magnitude at a single point near the tip')
      cf_chordwise_point_series, cf_chordwise_point_info = fl.cf_time_series_at_point(span_pct=span, chord_pct=chord, surface='Upper', component='chordwise')

      print('Point actually used: ', cf_chordwise_point_info)
      plot_cumulative_stats(
         cf_chordwise_point_series, dt=dt, rpm=rpm, sync='none', ylabel='$C_f$ [-]',
         savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_chordwise_point_cumulative_stats_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      print(40*'-')
      print('Plotting cumulative skewness+flatness of spanwise Cf')
      plot_cumulative_moments(
         cf_chordwise_point_series, dt=dt, rpm=rpm, sync='none', label='$C_{f,spanwise}$',
         savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_chordwise_point_cumulative_moments_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      print(40*'-')
      print('Plotting cf magnitude autocorrelation, first half vs second half of the run')
      plot_autocorrelation_windows(
      cf_chordwise_point_series, n_windows=2, dt=dt, labels=['First half', 'Second half'],
      savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_chordwise_point_autocorrelation_windows_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      print(40*'-')
      print('Plotting integral timescale / required averaging time for cf magnitude')
      plot_integral_timescale(
      cf_chordwise_point_series, dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
      savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_chordwise_point_integral_timescale_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      # Spanwise
      print(40*'-')
      print('Plotting cumulative mean+variance of Cf magnitude at a single point near the tip')
      cf_spanwise_point_series, cf_spanwise_point_info = fl.cf_time_series_at_point(span_pct=span, chord_pct=chord, surface='Upper', component='spanwise')

      print('Point actually used: ', cf_spanwise_point_info)
      plot_cumulative_stats(
         cf_spanwise_point_series, dt=dt, rpm=rpm, sync='none', ylabel='$C_f$ [-]',
         savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_spanwise_point_cumulative_stats_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      print(40*'-')
      print('Plotting cumulative skewness+flatness of spanwise Cf')
      plot_cumulative_moments(
         cf_spanwise_point_series, dt=dt, rpm=rpm, sync='none', label='$C_{f,spanwise}$',
         savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_spanwise_point_cumulative_moments_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      print(40*'-')
      print('Plotting cf magnitude autocorrelation, first half vs second half of the run')
      plot_autocorrelation_windows(
      cf_spanwise_point_series, n_windows=2, dt=dt, labels=['First half', 'Second half'],
      savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_spanwise_point_autocorrelation_windows_s{span:03d}_c{chord:03d}_{case}.png'),
      )

      print(40*'-')
      print('Plotting integral timescale / required averaging time for cf magnitude')
      plot_integral_timescale(
      cf_spanwise_point_series, dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
      savepath=os.path.join(master_path, f'images/cf/convergence/local/cf_spanwise_point_integral_timescale_s{span:03d}_c{chord:03d}_{case}.png'),
      )
