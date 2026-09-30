import os

from case_config import load_case_config
from bladeprocessor.strip_forces import StripForces
from bladeprocessor.convergence import plot_cumulative_stats
from bladeprocessor.convergence import plot_cumulative_mean
from bladeprocessor.convergence import plot_autocorrelation_windows
from bladeprocessor.convergence import plot_cumulative_moments
from bladeprocessor.convergence import standard_error
from bladeprocessor.convergence import plot_integral_timescale
from bladeprocessor.convergence import required_averaging_time
from bladeprocessor.convergence import plot_cycle_correlation

cfg = load_case_config()
master_path = cfg.master_path
case = cfg.case
inst_force_file = cfg.inst_force_file
avg_force_file = cfg.avg_force_file
r_tip = cfg.r_tip
rho_ref = cfg.rho_ref
rpm = cfg.rpm
span_axis = cfg.span_axis
chord_axis = cfg.chord_axis
thickness_axis = cfg.thickness_axis
validate_axes = cfg.validate_axes
span_min = cfg.span_min
dt = cfg.dt

for _sub in ('forces/convergence/global', 'forces/convergence/local', 'forces/hanson'):
	os.makedirs(os.path.join(master_path, 'images', _sub), exist_ok=True)


# ------------- Avg Strip forces (Hanson's method input) ------------- #

# Per-radial-strip, time-resolved axial/radial/tangential force, computed
# directly from a SNCReader.to_h5() file - replaces the manual PowerVIZ
# "Force Graph" CSV export (ForcesCSVConverter above). span_min isolates
# one blade (see README.md, "Strip forces" - same reason as everywhere
# else in this project). Check flip_axial/flip_tangential against what
# you expect physically before trusting the sign.

# print(40*'-')
# print('Opening StripForces file: ', os.path.join(master_path, avg_force_file))
# # span_min=span_min at load time (see FrictionLines' fl = ... in
# # skin_friction_manager.py for why) - every compute()/total_loads() call
# # below already passes span_min=span_min anyway to isolate one blade, so
# # this crops the force field actually read off disk to the same subset,
# # instead of loading the whole rotor.
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

# One harmonic at a time, around the true azimuth (see README.md,
# "One harmonic at a time, around the true azimuth: plot_harmonic_polar()")
# - a companion to the bar charts above, not a replacement: isolates a
# SINGLE chosen harmonic's own contribution and spells it out around a
# full revolution on a polar axis (radius = force, angle = azimuth),
# for one strip or a handful overlaid - NOT a substitute for
# reconstruct_from_harmonics()/plot_vs_angle()'s actual TOTAL loading
# curve (all harmonics summed). Pick the harmonic(s)/strip(s) actually
# worth a closer look at from the bar charts above first - 1P and the
# tip strip are typical starting points.
print(40*'-')
print('Plotting 1P polar contribution for the axial component, tip strip')
h_phase_axial = sf_inst.harmonics(result_inst, dt=dt, component='axial', n_harmonics=17, return_phase=True)
sf_inst.plot_harmonic_polar(
   h_phase_axial, harmonic=1, strips=9,
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonic_polar_1P_axial_{case}.png'),
)

print(40*'-')
print('Plotting 1P polar contribution for the axial component, several strips overlaid')
sf_inst.plot_harmonic_polar(
   h_phase_axial, harmonic=1, strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonic_polar_1P_overlay_axial_{case}.png'),
)

# Paired time-domain / harmonic-domain contour across the WHOLE span at
# once (see README.md, "Time + harmonic contour across the whole span:
# plot_harmonics_contour()") - a companion to the bar charts above, not a
# replacement: shows every strip at once instead of a handful, and pairs
# the raw unsteady-loading contour directly against its own harmonic
# content on a shared radius axis, so a feature visible in one (e.g. an
# impulsive event concentrated near the tip) can be read off directly
# against the other. Styled after Wu, Kingan, & Go (2022)'s own Fig. 20
# pairing - reuses the SAME h_phase_axial computed just above.
print(40*'-')
print('Plotting paired time/harmonic contour for the axial component')
sf_inst.plot_harmonics_contour(
   result_inst, h_phase_axial, dt=dt, component='axial',
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonics_contour_axial_{case}.png'),
)

print(40*'-')
print('Plotting paired time/harmonic contour for the tangential component')
sf_inst.plot_harmonics_contour(
   result_inst, h_phase_axial, dt=dt, component='tangential',
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonics_contour_tangential_{case}.png'),
)

print(40*'-')
print('Plotting paired time/harmonic contour for the radial component')
sf_inst.plot_harmonics_contour(
   result_inst, h_phase_axial, dt=dt, component='radial',
   savepath=os.path.join(master_path, f'images/forces/hanson/strip_harmonics_contour_radial_{case}.png'),
)


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
   savepath=os.path.join(master_path, f'images/forces/convergence/global/phase_portrait_axial_tangential_{case}.png'),
)

print(40*'-')
print('Plotting whole-blade phase portrait (axial vs radial)')
totals_inst = sf_inst.total_loads(span_min=span_min)  # standalone total, same span as result_inst above
sf_inst.plot_phase_portrait(
   totals_inst, component_pair=('axial', 'radial'),
   savepath=os.path.join(master_path, f'images/forces/convergence/global/phase_portrait_axial_radial_{case}.png'),
)

print(40*'-')
print('Plotting whole-blade phase portrait (radial vs tangential)')
totals_inst = sf_inst.total_loads(span_min=span_min)  # standalone total, same span as result_inst above
sf_inst.plot_phase_portrait(
   totals_inst, component_pair=('tangential', 'radial'),
   savepath=os.path.join(master_path, f'images/forces/convergence/global/phase_portrait_tangential_radial_{case}.png'),
)

print(40*'-')
print('Plotting per-strip phase portraits (axial vs radial) - localizes convergence issues by span')
sf_inst.plot_phase_portrait_by_strip(
   result_inst, component_pair=('axial', 'radial'), strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/convergence/global/phase_portrait_by_strip_axial_radial_{case}.png'),
)

print(40*'-')
print('Plotting per-strip phase portraits (axial vs tangential) - localizes convergence issues by span')
sf_inst.plot_phase_portrait_by_strip(
   result_inst, component_pair=('axial', 'tangential'), strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/convergence/global/phase_portrait_by_strip_axial_tangential_{case}.png'),
)

print(40*'-')
print('Plotting per-strip phase portraits (tangential vs radial) - localizes convergence issues by span')
sf_inst.plot_phase_portrait_by_strip(
   result_inst, component_pair=('tangential', 'radial'), strips=[0, 2, 4, 6, 8, 9],
   savepath=os.path.join(master_path, f'images/forces/convergence/global/phase_portrait_by_strip_tangential_radial_{case}.png'),
)

# Convergence checking: cumulative (running) mean vs revolutions included
# (see README.md, "Convergence checking: running/cumulative mean") - a
# converged quantity's running mean flattens to a horizontal asymptote.
# Not tied to StripForces specifically - takes any plain 1D array:
print(40*'-')
print('Plotting cumulative mean of thrust vs revolutions included')
plot_cumulative_mean(
   totals_inst['thrust'], dt=dt, rpm=rpm, ylabel='Thrust [N]',
   savepath=os.path.join(master_path, f'images/forces/convergence/global/thrust_cumulative_mean_{case}.png'),
)

print(40*'-')
print('Plotting cumulative mean of thrust vs revolutions included')
plot_cumulative_mean(
   totals_inst['torque'], dt=dt, rpm=rpm, ylabel='Torque [Nm]',
   savepath=os.path.join(master_path, f'images/forces/convergence/global/torque_cumulative_mean_{case}.png'),
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
   savepath=os.path.join(master_path, f'images/forces/convergence/global/thrust_cumulative_stats_{case}.png'),
)

print(40*'-')
print('Plotting cumulative mean+variance of thrust, synced to revolution boundaries')
plot_cumulative_stats(
   totals_inst['torque'], dt=dt, rpm=rpm, sync='none', ylabel='Torque [Nm]',
   savepath=os.path.join(master_path, f'images/forces/convergence/global/torque_cumulative_stats_{case}.png'),
)

# Example for a rotor-stator case instead (NOT this project's isolated
# rotor - shown for reference): 4 blades / 4 vanes repeat every
# 360/4=90 degrees, so sync every 90 degrees rather than every full
# revolution to get 4x the comparable-phase samples per run:
# plot_cumulative_stats(
#   totals_inst['thrust'], dt=dt, rpm=rpm, sync='periodicity', period_deg=90.0,
#   ylabel='Thrust [N]',
#   savepath=os.path.join(master_path, f'images/forces/convergence/global/thrust_cumulative_stats_periodicity_{case}.png'),
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
   savepath=os.path.join(master_path, f'images/forces/convergence/global/thrust_cumulative_moments_{case}.png'),
)

print(40*'-')
print('Plotting cumulative skewness+flatness of torque')
plot_cumulative_moments(
   totals_inst['torque'], dt=dt, rpm=rpm, sync='none', label='Torque',
   savepath=os.path.join(master_path, f'images/forces/convergence/global/torque_cumulative_moments_{case}.png'),
)

# # Rolling (fixed-size sliding window) overlay on the cumulative curve
# # above (see README.md, "Cross-checking the cumulative curve with a
# # fixed-size rolling window") - reveals a late-run drift a cumulative
# # curve's own shrinking (1/n) sensitivity can hide. window has no good
# # universal default - pick a few times integral_timescale()'s own
# # T_int (as a sample count) for this case, not a blindly reused number.
# print(40*'-')
# print('Plotting cumulative mean+variance of thrust with a rolling-window overlay')
# plot_cumulative_stats(
#    totals_inst['thrust'], dt=dt, rpm=rpm, sync='none', window=200, ylabel='Thrust [N]',
#    savepath=os.path.join(master_path, f'images/forces/convergence/global/thrust_cumulative_stats_rolling_{case}.png'),
# 

# Convergence checking: autocorrelation comparison between independent
# windows (see README.md, "Convergence checking: autocorrelation" - NOT
# a single-window "is rho(s) even" check, which is guaranteed to pass
# trivially regardless of convergence - comparing INDEPENDENT windows is
# what's actually meaningful):

print(40*'-')
print('Plotting thrust autocorrelation, first half vs second half of the run')
plot_autocorrelation_windows(
   totals_inst['thrust'], n_windows=2, dt=dt, labels=['First half', 'Second half'],
   savepath=os.path.join(master_path, f'images/forces/convergence/global/thrust_autocorrelation_windows_{case}.png'),
)

print(40*'-')
print('Plotting torque autocorrelation, first half vs second half of the run')
plot_autocorrelation_windows(
   totals_inst['torque'], n_windows=2, dt=dt, labels=['First half', 'Second half'],
   savepath=os.path.join(master_path, f'images/forces/convergence/global/torque_autocorrelation_windows_{case}.png'),
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
   savepath=os.path.join(master_path, f'images/forces/convergence/global/thrust_integral_timescale_{case}.png'),
)

plot_integral_timescale(
   totals_inst['torque'], dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
   savepath=os.path.join(master_path, f'images/forces/convergence/global/torque_integral_timescale_{case}.png'),
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
# harmonics() above and tip_vortex_manager.py's TipVortexPhaseAverage
# depend on. period_deg=360 (default) = one full revolution:
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


# Convergence at a SINGLE strip, not just the whole-blade total above
# (see README.md, "Convergence checking at a single point, not just the
# spatial mean") - a converged whole-blade total is necessary but not
# sufficient for convergence at any given strip; nearest_strip() picks
# one by r/R instead of an opaque strip index - 90% span (near the tip)
# is a reasonable "hardest to converge" starting choice, adjust per case.

span_pct = [25, 50, 75, 90]

for span in span_pct:
   # axial
   print(40*'-')
   print(f'Plotting cumulative mean+variance of axial force at the strip nearest {span}% span')
   tip_idx, tip_r_actual = sf_inst.nearest_strip(span, result_inst)
   print(f'Strip actually used: idx={tip_idx}, r={tip_r_actual:.4f} m')
   plot_cumulative_stats(
      result_inst['axial'][:, tip_idx], dt=dt, rpm=rpm, sync='none', ylabel='Axial force [N]',
      savepath=os.path.join(master_path, f'images/forces/convergence/local/axial_strip_cumulative_stats_s{span:03d}_{case}.png'),
   )

   print(f'Strip actually used: idx={tip_idx}, r={tip_r_actual:.4f} m')
   plot_cumulative_moments(
      result_inst['axial'][:, tip_idx], dt=dt, rpm=rpm, sync='none', label='Axial force',
      savepath=os.path.join(master_path, f'images/forces/convergence/local/axial_strip_cumulative_moments_s{span:03d}_{case}.png'),
   )

   print(40*'-')
   print('Plotting axial force autocorrelation, first half vs second half of the run')
   plot_autocorrelation_windows(
      result_inst['axial'][:, tip_idx], n_windows=2, dt=dt, labels=['First half', 'Second half'],
      savepath=os.path.join(master_path, f'images/forces/convergence/global/axial_strip_autocorrelation_windows_s{span:03d}_{case}.png'),
   )

   plot_integral_timescale(
      result_inst['axial'][:, tip_idx], dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
      savepath=os.path.join(master_path, f'images/forces/convergence/global/axial_strip_integral_timescale_s{span:03d}_{case}.png'),
   )

   # tangential
   print(40*'-')
   print(f'Plotting cumulative mean+variance of tangential force at the strip nearest {span}% span')
   print(f'Strip actually used: idx={tip_idx}, r={tip_r_actual:.4f} m')
   plot_cumulative_stats(
      result_inst['tangential'][:, tip_idx], dt=dt, rpm=rpm, sync='none', ylabel='Tangential force [N]',
      savepath=os.path.join(master_path, f'images/forces/convergence/local/tangential_strip_cumulative_stats_s{span:03d}_{case}.png'),
   )

   print(f'Strip actually used: idx={tip_idx}, r={tip_r_actual:.4f} m')
   plot_cumulative_moments(
      result_inst['tangential'][:, tip_idx], dt=dt, rpm=rpm, sync='none', label='Tangential force',
      savepath=os.path.join(master_path, f'images/forces/convergence/local/tangential_strip_cumulative_moments_s{span:03d}_{case}.png'),
   )

   print(40*'-')
   print('Plotting tangential force autocorrelation, first half vs second half of the run')
   plot_autocorrelation_windows(
      result_inst['tangential'][:, tip_idx], n_windows=2, dt=dt, labels=['First half', 'Second half'],
      savepath=os.path.join(master_path, f'images/forces/convergence/global/tangential_strip_autocorrelation_windows_s{span:03d}_{case}.png'),
   )

   plot_integral_timescale(
      result_inst['tangential'][:, tip_idx], dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
      savepath=os.path.join(master_path, f'images/forces/convergence/global/tangential_strip_integral_timescale_s{span:03d}_{case}.png'),
   )


   # radial
   print(40*'-')
   print(f'Plotting cumulative mean+variance of radial force at the strip nearest {span}% span')
   print(f'Strip actually used: idx={tip_idx}, r={tip_r_actual:.4f} m')
   plot_cumulative_stats(
      result_inst['radial'][:, tip_idx], dt=dt, rpm=rpm, sync='none', ylabel='Radial force [N]',
      savepath=os.path.join(master_path, f'images/forces/convergence/local/radial_strip_cumulative_stats_s{span:03d}_{case}.png'),
   )

   print(f'Strip actually used: idx={tip_idx}, r={tip_r_actual:.4f} m')
   plot_cumulative_moments(
      result_inst['radial'][:, tip_idx], dt=dt, rpm=rpm, sync='none', label='Radial force',
      savepath=os.path.join(master_path, f'images/forces/convergence/local/radial_strip_cumulative_moments_s{span:03d}_{case}.png'),
   )

   print(40*'-')
   print('Plotting radial force autocorrelation, first half vs second half of the run')
   plot_autocorrelation_windows(
      result_inst['radial'][:, tip_idx], n_windows=2, dt=dt, labels=['First half', 'Second half'],
      savepath=os.path.join(master_path, f'images/forces/convergence/global/radial_strip_autocorrelation_windows_s{span:03d}_{case}.png'),
   )

   plot_integral_timescale(
      result_inst['radial'][:, tip_idx], dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
      savepath=os.path.join(master_path, f'images/forces/convergence/global/radial_strip_integral_timescale_s{span:03d}_{case}.png'),
   )