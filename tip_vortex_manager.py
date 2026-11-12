import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from case_config import load_case_config
from bladeprocessor.meridional_field import (
    meridional_cylindrical_components, plot_meridional_field, scattered_cylindrical_components,
)
from bladeprocessor.vortex_core_tracking import (
    lamb_oseen, scully, vatistas_n2, taylor, plot_core_extraction_mesh,
    track_vortex_core_sweep, angle_at_wake_age, wake_age_series, vortex_profiles_at_angle,
    plot_core_trajectory, plot_circulation_vs_wakeage, plot_coreradius_vs_wakeage,
    plot_core_trajectory_overlay, plot_core_pressure_vs_wakeage, plot_core_pressure_coeff_vs_wakeage,
    plot_core_spread_vs_wakeage, plot_peak_swirl_vs_wakeage,
    plot_fig5_profiles, plot_fig6_profiles, plot_fig7_tangential_models,
    plot_fig8_coreradius_and_maxswirl_nd,
)

cfg = load_case_config()
master_path = cfg.master_path
r_tip = cfg.r_tip
rpm = cfg.rpm
pref = cfg.pref
rho_ref = cfg.rho_ref
normalize = cfg.normalize

# Dimensionless tip speed, used as normalize_by below only if this
# case's own yaml sets normalize: true - same convention pressure_manager.py/
# skin_friction_manager.py use for Cp/Cf (raw physical units when
# normalize is false, matching cases/*.yaml's own header comment).
u_tip = rpm * 2 * np.pi / 60 * r_tip

os.makedirs(os.path.join(master_path, 'images', 'tip_vortex'), exist_ok=True)
os.makedirs(os.path.join(master_path, 'images', 'meridional_field'), exist_ok=True)

# ------------- Tip-vortex tracking: scattered (x,y,z,value) phase-locked sweep (see README.md) ------------- #
#
# A separate dataset from everything above - 6e-5_6000rpm_PL/data/<component>/<name>_<N>deg_avg.txt,
# PowerVIZ's SCATTERED "position x y z value(<Variable>)" export (NOT the regular-grid Matrix
# format the section above reads - see meridional_field.py's own "Scattered (x, y, z, value)
# meridional exports" section for the file format and this dataset's own, separately-confirmed
# axis/rotation convention - do NOT assume it matches the Matrix-format dataset above; it doesn't
# have to and this one in particular genuinely sweeps the plane in 3D, confirmed via the
# thickness~0 check documented there, unlike the Matrix dataset's own fixed-plane/phase-locked
# finding).
#
# No cases/*.yaml entry for this dataset (yet) - it has no force/pressure h5 files at all, just
# this plane sweep, so r_tip/rpm are given directly here rather than forcing a placeholder case
# yaml's worth of unrelated required fields (inst_force, friction.radii, ...) that don't apply.

pl_master_path = '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm_PL'
# Same physical rotor as this case's own yaml (r_tip/rpm/pref/rho_ref all match
# cases/6e-5_6000rpm_HF.yaml exactly: 0.125 m, 6000 RPM, 101325 Pa, 1.22523 kg/m^3) - just a
# different CFD run/resolution with no yaml entry of its own (see comment above) - reuse the
# already-loaded cfg-derived constants directly instead of re-hardcoding separate copies that
# could silently drift from the yaml.
pl_r_tip = r_tip
pl_angles = list(range(0, 180, 10))  # 0,10,...,170deg - this dataset's own full set (see data/x-vel/)

os.makedirs(os.path.join(pl_master_path, 'images', 'tip_vortex_pl'), exist_ok=True)

for angle in pl_angles:
   angle_str = f'{angle:02d}'
   x_file = os.path.join(pl_master_path, f'data/x-vel/vel_x_{angle_str}deg_avg.txt')
   y_file = os.path.join(pl_master_path, f'data/y-vel/vel_y_{angle_str}deg_avg.txt')
   z_file = os.path.join(pl_master_path, f'data/z-vel/vel_z_{angle_str}deg_avg.txt')

   R, Y, v_r, v_a, v_t = scattered_cylindrical_components(x_file, y_file, z_file, angle_deg=float(angle))

   for prefix, field, sym in (('Ua', v_a, r'U_a'), ('Ur', v_r, r'U_r'), ('Ut', v_t, r'U_\theta')):
       plot_meridional_field(
           R, Y, field,
           variable_label=rf'Mean ${sym}$ [m/s]',
           r_tip=pl_r_tip, vmin=-18, vmax=18,
           savepath=os.path.join(pl_master_path, f'images/tip_vortex_pl/{prefix}_{angle:02d}deg.png'),
       )
   print(f'{angle_str}deg: done')


# ------------- Tip-vortex core tracking: location, circulation, core radius vs wake age ------------- #
#
# Methodology follows Quaglia, Leonard, Moreau, & Roger (2017), 10.1016/j.jsv.2017.03.023 (this
# project's own PI is a co-author) - see bladeprocessor/vortex_core_tracking.py's own module
# docstring for the full mapping between that paper's vortex-frame (r_v, theta_v, z_v) quantities
# and this dataset's own confirmed swirl-plane convention (v_swirl built from v_r/v_axial here,
# NOT v_theta - v_theta plays their "V_z", the along-core-axis component, instead). All the actual
# tracking/fitting/plotting logic lives in bladeprocessor/vortex_core_tracking.py (same split as
# every other *_manager.py here) - see each function's own docstring for what it does and why;
# this section is just the sequence of calls.
#
# IMPORTANT: track_angles is ordered DESCENDING (170 -> 0), NOT ascending like pl_angles above -
# 170deg is wake age 0 (t=0) - see track_vortex_core_sweep()'s own docstring for why (every
# single angle file already shows multiple discrete, ~0.2R-apart vortex cores from successive
# blade passages at once, and the ascending order was confirmed by direct inspection to make the
# continuity tracker jump between them instead of following one).

pl_u_tip = u_tip
track_angles = list(range(170, -10, -10))
rho_values = np.linspace(0.0005, 0.012, 30)  # 0.004R to 0.1R - capped under half the ~0.2R
# core-to-core spacing (confirmed by direct image inspection).

MODELS = {'lo': lamb_oseen, 'scully': scully, 'vatistas': vatistas_n2, 'taylor': taylor}
MODEL_LABELS = {'lo': 'Lamb-Oseen', 'scully': 'Scully', 'vatistas': 'Vatistas $n=2$', 'taylor': 'Taylor'}
MODEL_STYLES = {'lo': '-', 'scully': '--', 'vatistas': ':', 'taylor': '-.'}
q_tip = 0.5 * rho_ref * pl_u_tip ** 2

print(40 * '-')
print('Tracking tip-vortex core across the full wake-age sweep')
track_results, field_cache = track_vortex_core_sweep(
   pl_master_path, track_angles, pl_r_tip, rho_values, MODELS, MODEL_LABELS, MODEL_STYLES,
   profile_plot_dir=os.path.join(pl_master_path, 'images/tip_vortex_pl'),
   results_savepath=os.path.join(pl_master_path, 'images/tip_vortex_pl/vortex_core_results.npz'),
)

print(40 * '-')
print('Plotting core trajectory / circulation / core-radius vs wake age')
plot_core_trajectory(
   track_results, pl_r_tip,
   savepath=f'{pl_master_path}/images/tip_vortex_pl/core_trajectory.png',
)
plot_circulation_vs_wakeage(
   track_results, MODELS, MODEL_LABELS, MODEL_STYLES,
   savepath=f'{pl_master_path}/images/tip_vortex_pl/circulation_vs_wakeage.png',
)
plot_coreradius_vs_wakeage(
   track_results, MODELS, MODEL_LABELS, MODEL_STYLES,
   savepath=f'{pl_master_path}/images/tip_vortex_pl/coreradius_vs_wakeage.png',
)

# Combined "all cuts" overview (core trajectory on top of a representative meridional field) +
# independent diagnostics: core pressure minimum (raw + Cp-like), confidence spread, and peak
# swirl velocity (with its viscous-diffusion reference line) - see each function's own docstring.
angle_mid = angle_at_wake_age(track_results, 90)
plot_core_trajectory_overlay(
   track_results, field_cache, angle_mid, pl_r_tip,
   savepath=f'{pl_master_path}/images/tip_vortex_pl/core_trajectory_overlay.png',
)
plot_core_pressure_vs_wakeage(
   track_results, pref,
   savepath=f'{pl_master_path}/images/tip_vortex_pl/core_pressure_vs_wakeage.png',
)
plot_core_pressure_coeff_vs_wakeage(
   track_results, pref, q_tip,
   savepath=f'{pl_master_path}/images/tip_vortex_pl/core_pressure_coeff_vs_wakeage.png',
)
plot_core_spread_vs_wakeage(
   track_results,
   savepath=f'{pl_master_path}/images/tip_vortex_pl/core_spread_vs_wakeage.png',
)
plot_peak_swirl_vs_wakeage(
   track_results, rpm, pl_u_tip,
   savepath=f'{pl_master_path}/images/tip_vortex_pl/peak_swirl_vs_wakeage.png',
)

# Quaglia et al.'s own Fig. 4-8, replicated independently (pick whichever you actually want to
# use - see each function's own docstring). Fig 7b (axial-deficit Gaussian model) and Fig 9
# (radial velocity vs azimuth) are NOT implemented - no verified formula for 7b's Gaussian-deficit
# model, and Fig 9 needs modeling the vortex's own convection velocity, neither resolved with
# confidence - flagged as deferred rather than guessed.
print(40 * '-')
print('Plotting Quaglia et al. Fig. 4-8 replicas')
angle_late = angle_at_wake_age(track_results, 160)
angle_early = angle_at_wake_age(track_results, 10)
row_late = next(r for r in track_results if r['angle'] == angle_late)
core_late = field_cache[angle_late]['core']

plot_core_extraction_mesh(
   pl_master_path, float(angle_late), core_late['r_core'], core_late['y_core'],
   field_folder='pstat', field_prefix='p_stat', variable_label=r'Static pressure [Pa]',
   window=0.012,
   savepath_mesh=f'{pl_master_path}/images/tip_vortex_pl/fig4a_raw_mesh.png',
   savepath_polar=f'{pl_master_path}/images/tip_vortex_pl/fig4b_polar_mesh.png',
)

profiles_mid = vortex_profiles_at_angle(pl_master_path, field_cache, angle_mid, rho_values)
plot_fig5_profiles(
   rho_values, *profiles_mid[:4],
   savepath_tangential=f'{pl_master_path}/images/tip_vortex_pl/fig5a_tangential_profile.png',
   savepath_axial=f'{pl_master_path}/images/tip_vortex_pl/fig5b_axial_profile.png',
)

profiles_by_age = {
   'early': vortex_profiles_at_angle(pl_master_path, field_cache, angle_early, rho_values),
   'mid': profiles_mid,
   'late': vortex_profiles_at_angle(pl_master_path, field_cache, angle_late, rho_values),
}
wake_age_by_angle = dict(zip((row['angle'] for row in track_results), wake_age_series(track_results)))
age_labels = {k: f'{int(wake_age_by_angle[a])}' + r'$^\circ$'
              for k, a in (('early', angle_early), ('mid', angle_mid), ('late', angle_late))}
age_markers = {'early': 'x', 'mid': 'o', 'late': '+'}

plot_fig6_profiles(
   rho_values, pl_r_tip, pl_u_tip, pref, profiles_by_age, age_labels, age_markers,
   savepath_tangential=f'{pl_master_path}/images/tip_vortex_pl/fig6a_tangential_nd.png',
   savepath_axial_deficit=f'{pl_master_path}/images/tip_vortex_pl/fig6b_axial_deficit_nd.png',
   savepath_pressure=f'{pl_master_path}/images/tip_vortex_pl/fig6c_pressure_nd.png',
)

plot_fig7_tangential_models(
   rho_values, pl_r_tip, pl_u_tip, profiles_by_age['late'][0], row_late,
   MODELS, MODEL_LABELS, MODEL_STYLES, model_names=('lo', 'scully', 'taylor'),
   savepath=f'{pl_master_path}/images/tip_vortex_pl/fig7a_tangential_models.png',
)

plot_fig8_coreradius_and_maxswirl_nd(
   track_results, pl_r_tip, pl_u_tip,
   savepath_coreradius=f'{pl_master_path}/images/tip_vortex_pl/fig8a_coreradius_nd.png',
   savepath_maxswirl=f'{pl_master_path}/images/tip_vortex_pl/fig8b_maxswirl_nd.png',
)
