import os
import numpy as np
import matplotlib.pyplot as plt

from case_config import load_case_config
from bladeprocessor.blades_postproc import BladePostProcessor
from bladeprocessor.surface_field import SurfaceField, SurfaceFieldComparator
from bladeprocessor.surface_variable import SurfaceVariable
from bladeprocessor.convergence import plot_cumulative_stats
from bladeprocessor.convergence import plot_cumulative_mean
from bladeprocessor.convergence import plot_autocorrelation_windows
from bladeprocessor.convergence import plot_cumulative_moments
from bladeprocessor.convergence import plot_integral_timescale
from bladeprocessor.convergence import plot_cycle_correlation
from bladeprocessor.convergence import plot_periodogram
from bladeprocessor.convergence import plot_phase_lock
from bladeprocessor.pod import pod
from bladeprocessor.pod import plot_pod_mode
from bladeprocessor.pod import plot_pod_energy
from bladeprocessor.pod import plot_pod_temporal_coefficient

cfg = load_case_config()
master_path = cfg.master_path
case = cfg.case
avg_pressure_file = cfg.avg_pressure_file
inst_pressure_file = cfg.inst_pressure_file
r_tip = cfg.r_tip
rho_ref = cfg.rho_ref
rpm = cfg.rpm
pref = cfg.pref
c_ref = cfg.c_ref
normalize = cfg.normalize
span_axis = cfg.span_axis
chord_axis = cfg.chord_axis
thickness_axis = cfg.thickness_axis
span_min = cfg.span_min
reverse_chord = cfg.reverse_chord
radii = cfg.radii
dt = cfg.dt
blade_figsize = cfg.blade_figsize
frame_loop_step = cfg.frame_loop_step

for _sub in ('cp/avg', 'cp/inst', 'cp/rms', 'cp/convergence/global', 'cp/convergence/local', 'cp/pod', 'pfluct/spectra'):
	os.makedirs(os.path.join(master_path, 'images', _sub), exist_ok=True)

os.makedirs(os.path.join(master_path, 'data', 'pfluct'), exist_ok=True)


# ------------- Any surface variable at radii (Cp, y+, RMS, ...) ------------- #
#
# Input: a convert_snc_to_h5(..., surface_split=True) file - the pressure
# branch, for Cp (needs pf2ens's Static Pressure, see README.md, "2. Static
# Pressure"). Works just as well against a SNCReader.to_h5() forces-branch
# file for any variable stored there instead (Skin_Friction, y+ if
# present, etc. - see skin_friction_manager.py for the Skin_Friction
# case, and README.md, "Any surface variable at radii").

# print(40*'-')
# print('Opening SurfaceVariable file: ', os.path.join(master_path, avg_pressure_file))
# sv_pressure = SurfaceVariable(
#    os.path.join(master_path, avg_pressure_file),
#    r_tip=r_tip,
#    rho_ref=rho_ref,
#    rpm=rpm,
#    pref=pref,
#    c_ref=c_ref,
#    span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis
# )

# "Convergence checking on pressure") - needs an INSTANTANEOUS (multi-
# frame) pressure file, never a PowerFLOW-pre-averaged one, same
# requirement as everywhere else in this project:
print(40*'-')
print('Opening SurfaceVariable file: ', os.path.join(master_path, inst_pressure_file))
sv_pressure_inst = SurfaceVariable(
  os.path.join(master_path, inst_pressure_file),
  r_tip=r_tip, rho_ref=rho_ref, rpm=rpm, pref=pref, c_ref=c_ref,
  span_axis=span_axis, chord_axis=chord_axis, thickness_axis=thickness_axis
)

# # Raw access to any stored variable - instantaneous, mean, or rms/raw_rms:
#yplus_mean = sv.variable('y+', surface='Upper', frame=None, stat='mean')
#yplus_frame0 = sv.variable('y+', surface='Upper', frame=0)  # stat ignored once frame is set

# # Cp (same LOCAL q_ref normalization as FrictionLines.cf() - see README.md's
# # "Equations" section), one frame, the average, or its RMS fluctuation:
#cp_mean = sv.cp(surface='Upper', frame=None, stat='mean')
#cp_frame0 = sv.cp(surface='Upper', frame=0)
#cp_rms = sv.cp(surface='Upper', frame=None, stat='rms')

# # ------------- Convergence checking on pressure (see README.md) ------------- #

# # Spatial mean pressure/Cp per frame - same reduction
# # plot_cf_phase_portrait() uses on cf_time_series() in skin_friction_manager.py -
# # every convergence.py function takes this plain 1D per-frame series, exactly
# # like thrust/torque in forces_manager.py:
# if normalize:
#    # Cp-based series (pref subtracted, divided by LOCAL q_ref per point -
#    # see cp_time_series()) instead of raw pressure - hides the actual
#    # dynamic pressure scale (tied to r_tip/rpm) the same way normalize
#    # hides radius/chord elsewhere. cp_spatial_mean_time_series() (not
#    # cp_time_series().mean(axis=1)) - streams through the file a few
#    # frames at a time instead of loading the full (n_frames, n_points)
#    # array first, which OOM'd on a real DNS-resolution case (a (3674,
#    # 84898396) float32 array is 1.13 TiB; the spatial-mean result itself
#    # is only one scalar per frame).
#    p_series = sv_pressure_inst.cp_spatial_mean_time_series(surface='Upper')[:-2]  # drop corrupted last frame(s)
#    p_ylabel, p_label = '$C_p$ [-]', '$C_p$'
# else:
#    p_series = sv_pressure_inst.variable_spatial_mean_time_series('static_pressure', surface='Upper')[:-2]  # drop corrupted last frame(s)
#    p_ylabel, p_label = 'Pressure [Pa]', 'Pressure'

# print(40*'-')
# print('Plotting cumulative mean+variance of pressure')
# plot_cumulative_stats(
#   p_series, dt=dt, rpm=rpm, sync='none', ylabel=p_ylabel,
#   savepath=os.path.join(master_path, f'images/cp/convergence/global/pressure_cumulative_stats_{case}.png'),
# )

# print(40*'-')
# print('Plotting integral timescale / required averaging time for pressure')
# plot_integral_timescale(
#   p_series, dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
#   savepath=os.path.join(master_path, f'images/cp/convergence/global/pressure_integral_timescale_{case}.png'),
# )

# print(40*'-')
# print('Plotting cumulative mean of pressure vs revolutions included')
# plot_cumulative_mean(
#   p_series, dt=dt, rpm=rpm, ylabel=p_ylabel,
#   savepath=os.path.join(master_path, f'images/cp/convergence/global/pressure_cumulative_mean_{case}.png'),
# )

# print(40*'-')
# print('Plotting cumulative skewness+flatness of pressure')
# plot_cumulative_moments(
#   p_series, dt=dt, rpm=rpm, sync='none', label=p_label,
#   savepath=os.path.join(master_path, f'images/cp/convergence/global/pressure_cumulative_moments_{case}.png'),
# )

# print(40*'-')
# print('Plotting pressure autocorrelation, first half vs second half of the run')
# plot_autocorrelation_windows(
#   p_series, n_windows=2, dt=dt, labels=['First half', 'Second half'],
#   savepath=os.path.join(master_path, f'images/cp/convergence/global/pressure_autocorrelation_windows_{case}.png'),
# )

# # print(40*'-')
# # print('Plotting cycle-to-cycle correlation of pressure')
# # plot_cycle_correlation(
# #   p_series, dt=dt, rpm=rpm, period_deg=72,
# #   savepath=os.path.join(master_path, f'images/cp/convergence/global/pressure_cycle_correlation_{case}.png'),
# # )

# # Convergence at a SINGLE point, not just the spatial mean above (see
# # README.md, "Convergence checking at a single point, not just the
# # spatial mean") - a converged spatial mean is necessary but not
# # sufficient for convergence at any given point; span_pct/chord_pct=90/25
# # targets a point near the tip, a location expected to be among the
# # hardest to converge - see skin_friction_manager.py's own per-point loop
# # (cf_time_series_at_point) for the same reasoning applied to Cf; this is
# # the raw-pressure equivalent, via variable_time_series_at_point() rather
# # than a Cp-based one:
# chord_pts = np.arange(0, 101, 10)
# span_pts = [50, 70, 80, 90]

# for chord in chord_pts:
#    for span in span_pts:
#       print(40*'-')
#       print('Plotting cumulative mean+variance of pressure at a single point near the tip')
#       p_point_series, p_point_info = sv_pressure_inst.variable_time_series_at_point(
#          'static_pressure', span_pct=span, chord_pct=chord, surface='Upper')  # drop corrupted last frame(s)
#       p_point_series = p_point_series[:-2]

#       print('Point actually used: ', p_point_info)
#       plot_cumulative_stats(
#          p_point_series, dt=dt, rpm=rpm, sync='none', ylabel='Pressure [Pa]',
#          savepath=os.path.join(master_path, f'images/cp/convergence/local/pressure_point_cumulative_stats_s{span:03d}_c{chord:03d}_{case}.png'),
#       )

#       print(40*'-')
#       print('Plotting cumulative skewness+flatness of pressure at a single point')
#       plot_cumulative_moments(
#          p_point_series, dt=dt, rpm=rpm, sync='none', label='Pressure',
#          savepath=os.path.join(master_path, f'images/cp/convergence/local/pressure_point_cumulative_moments_s{span:03d}_c{chord:03d}_{case}.png'),
#       )

#       print(40*'-')
#       print('Plotting pressure autocorrelation at a single point, first half vs second half of the run')
#       plot_autocorrelation_windows(
#          p_point_series, n_windows=2, dt=dt, labels=['First half', 'Second half'],
#          savepath=os.path.join(master_path, f'images/cp/convergence/local/pressure_point_autocorrelation_windows_s{span:03d}_c{chord:03d}_{case}.png'),
#       )

#       print(40*'-')
#       print('Plotting integral timescale / required averaging time for pressure at a single point')
#       plot_integral_timescale(
#          p_point_series, dt=dt, rpm=rpm, sync='none', target_relative_sem=0.01,
#          savepath=os.path.join(master_path, f'images/cp/convergence/local/pressure_point_integral_timescale_s{span:03d}_c{chord:03d}_{case}.png'),
#       )


# # ------------- Radii cuts profiles for surface variables ------------- #

# # Cp vs local x/c at several radii, BOTH surfaces in one plot - span_min
# # isolates one blade half (see skin_friction_manager.py's friction_lines()
# # for why), and reverse_chord fixes which end is the leading vs. trailing
# # edge (no automatic detection - check per case, see the method's docstring):
# print(40*'-')
# print('Plotting Cp vs x/c at several radii, average over all frames')
# sv_pressure_inst.plot_cp_radii(
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
# # Generalizes friction_lines() (skin_friction_manager.py) to any scalar
# # field, and to_common_grid()/field() lets a SurfaceVariable slot into
# # the existing SurfaceField/SurfaceFieldComparator machinery for
# # cross-case deltas - see README.md, "Whole-blade surface plot" /
# # "Cross-case comparison".

# # Whole-blade -Cp scatter, both surfaces:
# print(40*'-')
# print('Plotting -Cp surface scatter, average over all frames')
# sv_pressure_inst.plot_variable_surface(
#    lambda s: -sv_pressure_inst.cp(surface=s, stat='mean'),
#    cbar_label='-Cp', span_min=span_min, surface='Upper',
#    figsize=blade_figsize,
#    normalize=normalize,
#    savepath=os.path.join(master_path, f'images/cp/avg/cp_surface_avg_upper_{case}.png'),
# )

# print(40*'-')
# print('Plotting -Cp surface scatter rms value')
# sv_pressure_inst.plot_variable_surface(
# lambda s: -sv_pressure_inst.cp(surface=s, stat='rms'),
# cbar_label='-Cp', span_min=span_min, surface='Upper',
# figsize=blade_figsize,
# normalize=normalize,
# savepath=os.path.join(master_path, f'images/cp/rms/cp_surface_rms_upper_{case}.png'),
# )

# for frame in range(0, sv_pressure_inst.n_frames, frame_loop_step):
#   print(40*'-')
#   print('Plotting -Cp surface scatter, instantaneous frame ', frame)
#   sv_pressure_inst.plot_variable_surface(
#     lambda s: -sv_pressure_inst.cp(surface=s, frame=frame),
#     cbar_label='-Cp', span_min=span_min, surface='Upper',
#     figsize=blade_figsize,
#     normalize=normalize,
#     savepath=os.path.join(master_path, f'images/cp/inst/cp_surface_upper_frame{frame:03d}_{case}.png'),
#   )

# # ------------- Computation with instatenous files containing pressure ------------- #

# # Pressure fluctuation p'(frame) = p(frame) - p_mean, one blade contour
# # per frame - needs a multi-frame file to show a real signal (a
# # single-frame file gives exactly 0 everywhere, since p(frame) == p_mean):
# for frame in range(0, sv_pressure_inst.n_frames, frame_loop_step):
#   print(40*'-')
#   print('Plotting pressure fluctuation, instantaneous frame ', frame)
#   sv_pressure_inst.plot_pressure_fluctuation(
#       frame, span_min=span_min, surface='Upper',
#       figsize=blade_figsize,
#       normalize=normalize,
#       savepath=os.path.join(master_path, f'images/pfluct/p_fluct_upper_frame{frame:03d}_{case}.png'),
#    )

# Prms needs no new method - it's already variable(stat='rms'). Normalized,
# this switches to Cp's own rms (Cp_rms = Prms / q_ref, LOCAL per-point
# normalization - same as cp()) instead of raw Prms, so the value itself
# is dimensionless too, not just the axes.
print(40*'-')
print('Plotting pressure RMS surface scatter, average over all frames')
if normalize:
   prms_get_values = lambda s: sv_pressure_inst.cp(surface=s, stat='rms')
   prms_cbar_label = '$C_{p,rms}$ [-]'
else:
   prms_get_values = lambda s: sv_pressure_inst.variable('static_pressure', surface=s, stat='rms')
   prms_cbar_label = '$P_{rms}$ [Pa]'
sv_pressure_inst.plot_variable_surface(
   prms_get_values,
   cbar_label=prms_cbar_label, span_min=span_min, surface='Upper',
   figsize=blade_figsize,
   normalize=normalize,
   savepath=os.path.join(master_path, f'images/pfluct/p_rms_surface_upper_{case}.png'),
)

# Point time trace + Welch periodogram (wall pressure fluctuations at one
# location, given as % of r/R and x/c - see README.md, "Point time trace").
# Needs a real time axis: pass dt explicitly if this file has no usable
# Metadata/mid_s (see README.md for when that's populated):

span_pcts = np.arange(70, 98, 2)  # % of r/R
chord_pcts = np.arange(0, 101, 10)  # % of x/c

for span in span_pcts:
  for chord in chord_pcts:
    print(40*'-')
    print(f'Plotting pressure time trace at span {span}% and chord {chord}%')
    sv_pressure_inst.plot_timetrace(
      'static_pressure', span_pct=span, chord_pct=chord, surface='Upper',
      ylabel='$C_p$ [-]' if normalize else 'Static pressure [Pa]', dt=dt,
      normalize=normalize,
      savepath=os.path.join(master_path, f'images/pfluct/spectra/p_timetrace_s{span:03d}_c{chord:03d}_{case}.png'),
    )
    print(40*'-')
    print(f'Plotting pressure periodogram at span {span}% and chord {chord}%')
    sv_pressure_inst.plot_periodogram(
      'static_pressure', span_pct=span, chord_pct=chord, surface='Upper', dt=dt,
      normalize=normalize, db=not normalize,
      savepath=os.path.join(master_path, f'images/pfluct/spectra/p_periodogram_s{span:03d}_c{chord:03d}_{case}.png'),
    )
    print(40*'-')
    print(f'Exporting pressure time trace at span {span}% and chord {chord}%')
    sv_pressure_inst.export_timetrace(
      'static_pressure', span_pct=span, chord_pct=chord, surface='Upper', dt=dt,
      savepath=os.path.join(master_path, f'data/pfluct/p_timetrace_s{span:03d}_c{chord:03d}_{case}.h5'),
    )
    # plot_timetrace()/plot_periodogram() each return (fig, ax) that's never
    # captured here, so each one stays open until the WHOLE script exits -
    # 10 span_pcts x 11 chord_pcts = 110 iterations x 2 plots/iteration =
    # 220 never-closed figures by the end of this loop alone, on top of
    # every OTHER plot already made above it. Confirmed to actually crash
    # (segfault, not a clean MemoryError) on a real large case - the
    # "More than 20 figures" RuntimeWarning earlier in the log was already
    # flagging exactly this. plt.close('all') is safe here since nothing
    # later needs an EARLIER iteration's figure still open.
    plt.close('all')

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

# ------------- Proper Orthogonal Decomposition (POD) of Cp ------------- #
#
# Objective, energy-ranked check of the spatial coherent structure
# identified from the pressure/separation analysis above (e.g. an
# LSB/leading-edge-vortex-like structure and its migration with span -
# see rotaris-docs/pod_section.tex for the full derivation). Computed on
# ONE surface at a time (Upper/suction side here, where that structure
# was identified - NOT both surfaces combined, matching how
# cp_time_series()/variable_time_series() are already surface-specific),
# restricted to the span window where the structure actually lives - NOT
# assumed to be the tip; pod_span_min/pod_span_max below are a
# placeholder and must be set from this case's own findings above before
# the result means anything.
#
# cp_time_series_region()/variable_time_series_region() (not
# cp_time_series()/variable_time_series() + a post-hoc column crop) -
# the crop is applied PER FRAME-CHUNK, during the streamed read itself,
# never holding the full (n_frames, n_points) array in memory at once
# (confirmed a real OOM on a DNS-resolution case otherwise - see those
# methods' docstrings). The OUTPUT is still held in full, though - if
# pod_span_min/pod_span_max below select too much of the blade for that
# to fit, the fix is a SMALLER region, not a different reading strategy;
# they're a placeholder and must be set from this case's own findings
# above before the result means anything either way.
p_label = '$C_p$' if normalize else 'Pressure'  # POD's own label - NOT dependent on the
# (commented-out above) convergence block's own p_label, which this section used to rely
# on - confirmed a real NameError once that block was commented out on its own.

pod_span_min, pod_span_max = 0.1 * r_tip, 1 * r_tip  # TODO: set from this case's own findings above
# pod_chord_min/pod_chord_max: set these (e.g. pod_chord_min = -0.02) if
# plot_pod_mode()'s own output shows a stray point sitting far from the
# rest of the selected region - confirmed a real degenerate/edge surfel
# on a real case, visible as a single dot that also drags the plot's own
# axis range out to accommodate it, squeezing the actual structure into
# a sliver. None (default, either one): no crop - check the printed
# chord min/max below first; only add a bound once you've confirmed
# there's a genuine single-point (or near-single-point) outlier, not
# real edge data (see FrictionLines/SurfaceVariable's own chord_percentile
# docstrings for the same caution about assuming something is an
# artifact rather than real physics).
pod_chord_min, pod_chord_max = None, None  # TODO: set from this case's own findings below, if needed

span_sv, chord_sv = sv_pressure_inst._span_chord('Upper')
radius_sv = sv_pressure_inst._radius('Upper')
# radius_sv (TRUE physical radius, [0, r_tip]) - NOT span_sv (the raw,
# CENTERED Cartesian span coordinate _span_chord() returns, roughly
# [-half_extent, +half_extent]) - pod_span_min/pod_span_max are fractions
# of r_tip, so they only mean "10%-100% of the blade" when compared
# against radius_sv. Comparing them against span_sv instead (an earlier
# version of this line did) silently selected some arbitrary narrow
# sliver near whichever end has positive centered-span values instead -
# confirmed on a real case (the Cf-POD equivalent in
# skin_friction_manager.py): 0.1*r_tip to 1*r_tip was intended to cover
# nearly the whole blade, but the plotted region only ever showed the
# last ~15% near the tip.
pod_region = (radius_sv >= pod_span_min) & (radius_sv <= pod_span_max)
print(f'pod_region chord range BEFORE any pod_chord_min/pod_chord_max crop: '
      f'[{chord_sv[pod_region].min():.4f}, {chord_sv[pod_region].max():.4f}] '
      f'({int(pod_region.sum())} points) - check this for a stray outlier before trusting the plot/POD below')
if pod_chord_min is not None:
   pod_region &= chord_sv >= pod_chord_min
if pod_chord_max is not None:
   pod_region &= chord_sv <= pod_chord_max

# Pressure-branch files (converters/ensight_to_h5.py) don't currently
# carry a per-surfel Area (unlike the forces branch - see
# SurfaceVariable._area()'s docstring), so fall back to an unweighted
# POD on this file rather than crash; the forces-branch Cf POD in
# skin_friction_manager.py is unaffected and stays area-weighted.
if sv_pressure_inst.areas['Upper'] is not None:
   pod_weights = sv_pressure_inst._area('Upper')[pod_region]
else:
   print(f"No per-surfel Area in '{inst_pressure_file}' (pressure-branch file) - running an UNWEIGHTED POD.")
   pod_weights = None

print(40*'-')
print(f'Running POD on {p_label}, Upper surface, {int(pod_region.sum())} points in the selected region')
if normalize:
   cp_pod_data = sv_pressure_inst.cp_time_series_region(surface='Upper', region_mask=pod_region)[:-2]  # drop corrupted last frame(s)
else:
   cp_pod_data = sv_pressure_inst.variable_time_series_region('static_pressure', surface='Upper', region_mask=pod_region)[:-2]  # drop corrupted last frame(s)
cp_pod_result = pod(cp_pod_data, n_modes=10, weights=pod_weights)
print(f'{p_label} POD energy fractions (first 5): ', cp_pod_result['energy_fraction'][:5])

print(40*'-')
print(f'Plotting {p_label} POD energy spectrum')
plot_pod_energy(
   cp_pod_result,
   savepath=os.path.join(master_path, f'images/cp/pod/cp_pod_energy_{case}.png'),
)

print(40*'-')
print(f'Plotting {p_label} POD mode 1 (most energetic) on the actual blade geometry')
plot_pod_mode(
   radius_sv[pod_region], chord_sv[pod_region], cp_pod_result, mode_index=0,
   unit_label=f'{p_label} mode amplitude [-]',
   figsize=blade_figsize,
   savepath=os.path.join(master_path, f'images/cp/pod/cp_pod_mode1_{case}.png'),
)

print(40*'-')
print(f'Plotting {p_label} POD mode 1 temporal coefficient')
plot_pod_temporal_coefficient(
   cp_pod_result, mode_index=0, dt=dt,
   savepath=os.path.join(master_path, f'images/cp/pod/cp_pod_mode1_coeff_{case}.png'),
)

# Is mode 1's own temporal coefficient a_1(t) tied to the rotor's own
# rotation (a peak at an integer multiple of f_rot, and/or a clean
# phase-locked curve vs azimuth), or does it carry an independent
# frequency of its own - see rotaris-docs/pod_section.tex's own
# discussion of this in relation to the harmonic decomposition, and the
# equivalent check already done on the Cf POD mode in
# skin_friction_manager.py.
print(40*'-')
print(f'Plotting {p_label} POD mode 1 temporal coefficient PSD')
plot_periodogram(
   cp_pod_result['coefficients'][:, 0], dt=dt,
   ylabel='$a_1(t)$ PSD',
   savepath=os.path.join(master_path, f'images/cp/pod/cp_pod_mode1_coeff_psd_{case}.png'),
)

print(40*'-')
print(f'Plotting {p_label} POD mode 1 temporal coefficient, phase-locked to rotor azimuth')
plot_phase_lock(
   cp_pod_result['coefficients'][:, 0], dt, rpm,
   ylabel='$a_1(t)$ phase-locked mean',
   savepath=os.path.join(master_path, f'images/cp/pod/cp_pod_mode1_coeff_phaselock_{case}.png'),
)

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


