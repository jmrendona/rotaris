'''
Cross-SIMULATION comparison driver - the "manager" for every plot that
overlays several separate cases on one figure (bladeprocessor/
case_comparison.py), as opposed to manager.py, which post-processes ONE
case at a time.

Motivating example (see bladeprocessor/case_comparison.py's own module
docstring for the full design rationale): a thesis results chapter
comparing several simulation variants of the same isolated hovering
rotor (e.g. this project's own '2025', '2025-T', '2026', '2026-DNS'
comparison) - a naive one-panel-per-case grid either buries the actual
cross-case comparison inside a dozen small, separately-legended panels,
or becomes unreadable once printed at a thesis page's actual physical
size. case_comparison.py's plot_time_trace_compare()/
plot_vs_angle_compare() fix that by putting every case on the SAME
panel (color = strip, linestyle = case, one shared legend) - this file
is where those get called for a specific, real set of case files.

Fill in CASES below with your own real file paths/parameters before
running - the values here are placeholders (this project's own
6e-5_6000rpm_HF case's rotor geometry, reused as a reasonable starting
point since these comparison cases are typically the same rotor at
different mesh/model fidelities, not a different rotor). Relative paths
in CASES are resolved against THIS file's own directory, same convention
as manager.py.
'''

import os
import numpy as np

from bladeprocessor.strip_forces import StripForces
from bladeprocessor.case_comparison import plot_time_trace_compare, plot_vs_angle_compare, plot_radii_compare
from bladeprocessor.friction_lines import FrictionLines
from bladeprocessor.surface_variable import SurfaceVariable
from bladeprocessor.meridional_field import scattered_cylindrical_components, griddata_scattered_field
from bladeprocessor.vortex_core_tracking import (
    row_at_wake_age, azimuthal_profiles, azimuthal_scalar_profile, lamb_oseen, scully, taylor,
    plot_core_trajectory_compare, plot_fig6_compare, plot_fig7_compare,
)

# ------------- Case configuration - FILL IN before running ------------- #
#
# One entry per simulation to compare, in the order they should appear
# in the shared legend and be assigned a linestyle
# (solid/dashed/dotted/dash-dot, cycling - see
# bladeprocessor.case_comparison._LINESTYLES). Every case is assumed to
# share the SAME rotor geometry (r_tip, rpm) and the SAME strip layout
# (n_span_bins, span_min/span_max) - required for plot_*_compare()'s
# shared `strips` index list to mean the same physical radius in every
# case (see case_comparison._check_consistent_strip_layout(), which
# warns, but does not raise, if that turns out not to hold).

MASTER_PATH = '/scratch/jmrendon/Rotor-alone/comparison'  # TODO: real output path for saved figures

CASES = {
    '2025':     {'inst_force_file': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm/2025_forces_rotor.h5',       # TODO
                 'inst_pressure_file': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm/2025_pressure_rotor.h5'},
    '2025-T':   {'inst_force_file': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm-transition/2025T_forces_rotor.h5',    # TODO
                 'inst_pressure_file': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm-transition/2025T_pressure_rotor.h5'},
    '2026':     {'inst_force_file': '/scratch/jmrendon/Rotor-alone/15e-6_6000rpm/2026_forces_rotor.h5',       # TODO
                 # No instantaneous Cp file converted for this case - using the AVG (single-frame)
                 # file instead: cp(stat='mean') on a 1-frame file just returns that one frame's
                 # own value, which already IS the mean.
                 'inst_pressure_file': '/scratch/jmrendon/Rotor-alone/15e-6_6000rpm/2026_avg_pressure_rotor.h5'},
    '2026-DNS': {'inst_force_file': '/scratch/jmrendon/Rotor-alone/1e-5_6000rpm-dns/2026DNS_forces_rotor.h5',  # TODO
                 'inst_pressure_file': '/scratch/jmrendon/Rotor-alone/1e-5_6000rpm-dns/2026DNS_pressure_rotor.h5'},
}

# Shared rotor/geometry/crop parameters - see cases/6e-5_6000rpm_HF.yaml
# for what each of these means; override per-case below (e.g. CASES
# ['2026-DNS']['rpm'] = ...) only if one case genuinely differs.
R_TIP = 0.125
RPM = 6000
DT = 0.0000556
SPAN_MIN = 0.02
SPAN_AXIS, CHORD_AXIS, THICKNESS_AXIS = 0, 2, 1
VALIDATE_AXES = False
N_SPAN_BINS = 10  # matches this project's own 10-equally-spaced-strips convention (see thesis Fig. 5.6/5.7)

# Which 3 (of N_SPAN_BINS) strips to compare - 0-indexed, root/mid/tip by
# default. Kept small deliberately (see case_comparison.py's own module
# docstring on why 10 strips x 4 cases overlaid on one panel would be
# unreadable) - adjust to whichever stations are most informative for
# this case (e.g. move the "tip" entry closer to N_SPAN_BINS - 1 if a
# near-tip effect - blade-vortex interaction, tip loss - is the point of
# interest, rather than plain root/mid/tip coverage).
STRIPS = (0, N_SPAN_BINS // 2, N_SPAN_BINS - 1)

os.makedirs(os.path.join(MASTER_PATH, 'images/forces/comparison'), exist_ok=True)
os.makedirs(os.path.join(MASTER_PATH, 'images/comparison/radii'), exist_ok=True)


# def _resolve(path):
#     '''Relative CASES paths are resolved against this file's own directory - same convention as manager.py.'''
#     if os.path.isabs(path):
#         return path
#     return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


# def load_case(case_label, case_cfg):

#     '''
#     Build one case's StripForces instance, its compute() result (for the
#     raw time-trace comparison), and its phase_lock() result (for the
#     phase-locked/polar comparison) - the two inputs plot_time_trace_compare()/
#     plot_vs_angle_compare() need, one dict entry per case.
#     '''

#     print(40 * '-')
#     print(f"Loading case '{case_label}': {case_cfg['inst_force_file']}")

#     sf = StripForces(
#         _resolve(case_cfg['inst_force_file']),
#         r_tip=case_cfg.get('r_tip', R_TIP), rpm=case_cfg.get('rpm', RPM),
#         span_min=case_cfg.get('span_min', SPAN_MIN),
#         span_axis=case_cfg.get('span_axis', SPAN_AXIS), chord_axis=case_cfg.get('chord_axis', CHORD_AXIS),
#         thickness_axis=case_cfg.get('thickness_axis', THICKNESS_AXIS),
#         validate_axes=case_cfg.get('validate_axes', VALIDATE_AXES),
#     )

#     result = sf.compute(span_min=case_cfg.get('span_min', SPAN_MIN), n_span_bins=case_cfg.get('n_span_bins', N_SPAN_BINS))
#     phase_locked = sf.phase_lock(result, dt=case_cfg.get('dt', DT), n_azimuth_bins=72)

#     return result, phase_locked


# def main():

#     results = {}
#     phase_locked_all = {}

#     for case_label, case_cfg in CASES.items():
#         result, phase_locked = load_case(case_label, case_cfg)
#         results[case_label] = result
#         phase_locked_all[case_label] = phase_locked

#     print(40 * '-')
#     print('Plotting cross-case raw time-trace comparison (Fig. 5.6 redesign)')
#     plot_time_trace_compare(
#         results, dt=DT, strips=STRIPS,
#         savepath=os.path.join(MASTER_PATH, 'images/forces/comparison/force_time_trace_compare.png'),
#     )

#     print(40 * '-')
#     print('Plotting cross-case phase-locked/polar comparison (Fig. 5.7 redesign)')
#     plot_vs_angle_compare(
#         phase_locked_all, strips=STRIPS,
#         savepath=os.path.join(MASTER_PATH, 'images/forces/comparison/force_vs_angle_compare.png'),
#     )


# ------------- Cross-SIMULATION Cp/Cf-vs-x/c comparison, at a few fixed radii ------------- #
#
# Overlays Cp and each Cf component (magnitude, chordwise, spanwise) vs. local x/c, at a handful
# of fixed radii, across every case in CASES above - the cross-case counterpart to
# SurfaceVariable.plot_cp_radii()/FrictionLines.plot_cf_radii() (which compare RADII within ONE
# case - see pressure_manager.py's/skin_friction_manager.py's own commented-out calls to those).
# Uses the SAME CASES dict as main()'s own StripForces comparison above (now also carrying each
# case's own inst_pressure_file - fill in/verify before running, same as inst_force_file).

REVERSE_CHORD = True  # TODO: verify per case - see FrictionLines.cf_at_radii()'s own docstring
RADII_COMPARE = (0.045, 0.100, 0.122)  # root/mid/tip of this project's own 5-radii convention
# (cases/*.yaml's own friction.radii) - kept to 3 for the same reason STRIPS is kept to 3 (see
# that constant's own comment): _LINESTYLES only has 4 visually-distinct entries, and more than a
# handful of (case, radius) combinations on one panel stops being readable. Cf only - Cp has its
# own RADII_COMPARE_CP below (different radii, to match the iLES reference data).

RHO_REF = 1.22523  # TODO: copy from cases/*.yaml's own rotor.rho_ref for this rotor
PREF = 101325.0  # TODO: copy from cases/*.yaml's own rotor.pref - needed for mean Cp (cp())

# Cp-specific radii (50/80/98% span, span=R_TIP=0.125 m) - chosen to match the 3 iLES reference
# stations below exactly, rather than this project's own usual root/mid/tip picks (RADII_COMPARE
# above), since the whole point of the Cp comparison is to overlay the LBM cases against iLES at
# the SAME physical locations.
RADII_COMPARE_CP = tuple(pct / 100 * R_TIP for pct in (50, 80, 98))

# Digitized iLES reference Cp data, given in RAW +Cp (confirmed directly by whoever supplied it -
# load_iles_cp_curves() negates it to match this project's own -Cp sign convention, see
# plot_cp_radii()'s own docstring) at the SAME 3 span stations as RADII_COMPARE_CP above
# (50/80/98%). Each file is one continuous digitized path around the airfoil (x/c increasing then
# decreasing - NOT separately split into Upper/Lower, and deliberately plotted in file order,
# never re-sorted by x/c, to avoid guessing at a turning point that isn't needed - see
# load_iles_cp_curves()).
ILES_FILES = {
    50: '/scratch/jmrendon/Rotor-alone/comparison/iLES_50pr.csv',
    80: '/scratch/jmrendon/Rotor-alone/comparison/iLES_80pr.csv',
    98: '/scratch/jmrendon/Rotor-alone/comparison/iLES_98pr.csv',
}


def load_iles_cp_curves():

    '''
    Loads the 3 digitized iLES +Cp-vs-x/c CSVs (ILES_FILES, header "x, y"), negating the raw
    values to -Cp (see ILES_FILES' own comment), into the same {radius: {surface: (xc, values)}}
    shape plot_radii_compare() expects for every other case -
    keyed by the SAME physical radii as RADII_COMPARE_CP (matched by percent), with a single
    'iLES' surface key per radius holding the ENTIRE digitized path as one (x, y) pair, in
    original file order (not split into Upper/Lower, not re-sorted - see ILES_FILES' own
    comment). Meant to be plotted with marker_only_cases=('iLES',) (see plot_radii_compare()),
    not as a connected line alongside the simulation cases.
    '''

    percent_to_radius = dict(zip((50, 80, 98), RADII_COMPARE_CP))
    curves = {}
    for percent, path in ILES_FILES.items():
        xy = np.genfromtxt(path, delimiter=',', skip_header=1)
        curves[percent_to_radius[percent]] = {'iLES': (xy[:, 0], -xy[:, 1])}  # +Cp as given -> -Cp
    return curves

# REQUIRED on this project's own multi-blade .snc/.fnc files - see cases/*.yaml's own crop:
# azimuth_center_deg/azimuth_tol_deg and bladeprocessor._azimuth's own docstring: span_min ALONE
# does not reliably isolate one blade (confirmed earlier on this project's own meshes - see
# README.md/case history). Omitting this crop here isn't just a correctness bug (both blades'
# chord ranges getting mixed into each radius band - see cf_at_radii()'s/at_radii()'s own
# docstrings), it's also a MEMORY bug specifically for FrictionLines: unlike SurfaceVariable
# (which streams Data in chunks), FrictionLines._load() eagerly loads the ENTIRE per-frame force
# field (shape (n_frames, n_selected_points, 3)) into memory at construction time - confirmed
# directly: omitting this crop OOM-killed a real run against this project's own 2025 case
# (~680 GB file, both blades x thousands of frames x 3 components, float64 - easily into the TB
# range uncropped). ALWAYS set this before running against a real multi-blade file.
AZIMUTH_CENTER_DEG = 0  # TODO: verify per case - see cases/*.yaml's own crop.azimuth_center_deg
AZIMUTH_TOL_DEG = 45  # TODO: verify per case - see cases/*.yaml's own crop.azimuth_tol_deg


def load_case_for_radii_compare(case_label, case_cfg):

    '''
    Builds one case's Cp (BOTH surfaces - see below, at RADII_COMPARE_CP) and Cf-component
    (magnitude/chordwise/spanwise, Upper surface only, at RADII_COMPARE) curves, mean over all
    frames - the inputs plot_radii_compare() needs, one dict entry per case. Returns None in
    place of the Cp curves
    if case_cfg has no 'inst_pressure_file' (e.g. a case with no instantaneous Cp conversion yet
    - see CASES' own '2026' entry) - that case is then simply skipped for the Cp comparison
    specifically, without blocking the Cf comparisons.

    Cp needs BOTH Upper AND Lower, unlike Cf (Upper alone is the point of interest there, e.g.
    separation/transition on the suction side) - a suction-side-only Cp-vs-x/c curve is not a
    physically complete picture (no Kutta-condition trailing-edge convergence from the pressure
    side, no pressure-side branch at all) - confirmed directly: an Upper-only cross-case Cp plot
    looked nothing like the matching single-case plot_cp_radii() reference plot, which always
    shows both surfaces in the same color per radius (see SurfaceVariable.plot_at_radii()'s own
    docstring) - exactly the convention plot_radii_compare() itself now follows too.
    '''

    azimuth_center_deg = case_cfg.get('azimuth_center_deg', AZIMUTH_CENTER_DEG)
    azimuth_tol_deg = case_cfg.get('azimuth_tol_deg', AZIMUTH_TOL_DEG)

    print(40 * '-')
    print(f"Loading Cf for case '{case_label}': {case_cfg['inst_force_file']}")
    fl = FrictionLines(
        case_cfg['inst_force_file'],
        r_tip=case_cfg.get('r_tip', R_TIP), rho_ref=case_cfg.get('rho_ref', RHO_REF), rpm=case_cfg.get('rpm', RPM),
        span_axis=case_cfg.get('span_axis', SPAN_AXIS), chord_axis=case_cfg.get('chord_axis', CHORD_AXIS),
        thickness_axis=case_cfg.get('thickness_axis', THICKNESS_AXIS),
        validate_axes=case_cfg.get('validate_axes', VALIDATE_AXES),
        azimuth_center_deg=azimuth_center_deg, azimuth_tol_deg=azimuth_tol_deg,
    )
    span_min = case_cfg.get('span_min', SPAN_MIN)
    reverse_chord = case_cfg.get('reverse_chord', REVERSE_CHORD)
    # Wrapped in a single-key {'Upper': ...} dict per radius - plot_radii_compare() now always
    # expects {radius: {surface: (xc, values)}} uniformly (Cp below needs 2 surface keys, Cf
    # only ever needs this 1 - see that function's own docstring).
    cf_mag = {r: {'Upper': xc_v} for r, xc_v in
              fl.cf_at_radii(RADII_COMPARE, surface='Upper', component=None,
                              span_min=span_min, reverse_chord=reverse_chord).items()}
    cf_chordwise = {r: {'Upper': xc_v} for r, xc_v in
                     fl.cf_at_radii(RADII_COMPARE, surface='Upper', component='chordwise',
                                     span_min=span_min, reverse_chord=reverse_chord).items()}
    cf_spanwise = {r: {'Upper': xc_v} for r, xc_v in
                    fl.cf_at_radii(RADII_COMPARE, surface='Upper', component='spanwise',
                                    span_min=span_min, reverse_chord=reverse_chord).items()}

    cp_curves = None
    if case_cfg.get('inst_pressure_file'):
        print(40 * '-')
        print(f"Loading Cp for case '{case_label}': {case_cfg['inst_pressure_file']}")
        sv = SurfaceVariable(
            case_cfg['inst_pressure_file'],
            r_tip=case_cfg.get('r_tip', R_TIP), rho_ref=case_cfg.get('rho_ref', RHO_REF),
            rpm=case_cfg.get('rpm', RPM), pref=case_cfg.get('pref', PREF),
            span_axis=case_cfg.get('span_axis', SPAN_AXIS), chord_axis=case_cfg.get('chord_axis', CHORD_AXIS),
            thickness_axis=case_cfg.get('thickness_axis', THICKNESS_AXIS),
            azimuth_center_deg=azimuth_center_deg, azimuth_tol_deg=azimuth_tol_deg,
        )
        # -Cp, not Cp: this project's own established sign convention for a mean-Cp-vs-x/c plot
        # (see SurfaceVariable.plot_cp_radii()'s own docstring - 'mean' plots -Cp so the suction
        # side's peak reads as a peak going UP, not down) - the first version of this comparison
        # plotted raw +Cp under a "$C_p$" label, which is what made it look wrong next to every
        # single-case reference plot (all of which use plot_cp_radii(), i.e. -Cp).
        cp_curves = sv.at_radii(RADII_COMPARE_CP, lambda s: -sv.cp(surface=s, stat='mean'),
                                 surface=('Upper', 'Lower'), span_min=span_min, reverse_chord=reverse_chord)
    else:
        print(40 * '-')
        print(f"Skipping Cp for case '{case_label}' - no 'inst_pressure_file' set.")

    return cp_curves, cf_mag, cf_chordwise, cf_spanwise


RADII_COMPARE_CACHE = os.path.join(MASTER_PATH, 'images/comparison/radii/radii_compare_cache.npz')
# Saved by main_radii_compare() below right after the expensive FrictionLines/SurfaceVariable
# extraction (the part that needed a 40-core/767GB compute-node job, not the plotting itself) -
# replot_radii_compare_from_cache() reloads this and just re-calls plot_radii_compare(), so a
# PURE STYLING/label change (like the color=radius/linestyle=case swap and the Cp sign fix this
# cache mechanism was added alongside) never needs to pay for that extraction again.


def main_radii_compare():

    cp_by_case, cf_mag_by_case, cf_chordwise_by_case, cf_spanwise_by_case = {}, {}, {}, {}

    for case_label, case_cfg in CASES.items():
        cp_curves, cf_mag, cf_chordwise, cf_spanwise = load_case_for_radii_compare(case_label, case_cfg)
        if cp_curves is not None:
            cp_by_case[case_label] = cp_curves
        cf_mag_by_case[case_label] = cf_mag
        cf_chordwise_by_case[case_label] = cf_chordwise
        cf_spanwise_by_case[case_label] = cf_spanwise

    print(40 * '-')
    print(f'Caching extracted curves to {RADII_COMPARE_CACHE}')
    np.savez(RADII_COMPARE_CACHE, allow_pickle=True,
             cp_by_case=cp_by_case, cf_mag_by_case=cf_mag_by_case,
             cf_chordwise_by_case=cf_chordwise_by_case, cf_spanwise_by_case=cf_spanwise_by_case)

    plot_radii_compare_results(cp_by_case, cf_mag_by_case, cf_chordwise_by_case, cf_spanwise_by_case)


def replot_radii_compare_from_cache(cache_path: str = RADII_COMPARE_CACHE):

    '''
    Re-plots the 4 radii-comparison figures from main_radii_compare()'s own cached curves (see
    RADII_COMPARE_CACHE's own comment) - no FrictionLines/SurfaceVariable re-extraction, so this
    runs in seconds and is safe to call interactively/on a login node, unlike main_radii_compare()
    itself. Use this for any plot_radii_compare()-level tweak (colors, styling, labels) instead of
    resubmitting the full extraction job.
    '''

    data = np.load(cache_path, allow_pickle=True)
    plot_radii_compare_results(
        data['cp_by_case'].item(), data['cf_mag_by_case'].item(),
        data['cf_chordwise_by_case'].item(), data['cf_spanwise_by_case'].item(),
    )


def plot_radii_compare_results(cp_by_case, cf_mag_by_case, cf_chordwise_by_case, cf_spanwise_by_case):

    print(40 * '-')
    print('Plotting cross-case Cp vs x/c comparison (+ iLES reference)')
    if cp_by_case:
        cp_by_case_with_iles = {**cp_by_case, 'iLES': load_iles_cp_curves()}
        plot_radii_compare(
            cp_by_case_with_iles, ylabel=r'$-C_p$ [-]', marker_only_cases=('iLES',),
            savepath=os.path.join(MASTER_PATH, 'images/comparison/radii/cp_radii_compare.png'),
        )
    else:
        print('  (skipped - no case had an inst_pressure_file set)')

    print(40 * '-')
    print('Plotting cross-case Cf (magnitude) vs x/c comparison')
    plot_radii_compare(
        cf_mag_by_case, ylabel=r'$|C_f|$ [-]',
        savepath=os.path.join(MASTER_PATH, 'images/comparison/radii/cf_magnitude_radii_compare.png'),
    )

    print(40 * '-')
    print('Plotting cross-case Cf (chordwise) vs x/c comparison')
    plot_radii_compare(
        cf_chordwise_by_case, ylabel=r'$C_f$ (chordwise) [-]',
        savepath=os.path.join(MASTER_PATH, 'images/comparison/radii/cf_chordwise_radii_compare.png'),
    )

    print(40 * '-')
    print('Plotting cross-case Cf (spanwise) vs x/c comparison')
    plot_radii_compare(
        cf_spanwise_by_case, ylabel=r'$C_f$ (spanwise) [-]',
        savepath=os.path.join(MASTER_PATH, 'images/comparison/radii/cf_spanwise_radii_compare.png'),
    )


# ------------- Cross-SIMULATION tip-vortex (TV) tracking comparison ------------- #
#
# Overlays core_trajectory, Fig. 6 (tangential/axial-deficit/pressure, dimensionless), and Fig. 7a
# (tangential + model fits) across several TV tracking runs - the cross-case counterpart to
# tip_vortex_manager.py's own single-case versions (bladeprocessor.vortex_core_tracking's
# plot_core_trajectory()/plot_fig6_profiles()/plot_fig7_tangential_models()). See
# plot_core_trajectory_compare()'s/plot_fig6_compare()'s/plot_fig7_compare()'s own docstrings for
# the color=category/linestyle=case convention.
#
# A SEPARATE dict from CASES above - a TV tracking run lives in its own PL dataset directory (see
# tip_vortex_manager.py's own module comment on this), not the inst_force_file/inst_pressure_file
# pair CASES uses, and not every case necessarily HAS a completed TV tracking run. The two entries
# below are this project's own actual completed runs so far (see tip_vortex_manager.py) - add
# more as additional runs get tracked.

TV_CASES = {
    'HF':     {'pl_master_path': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm_PL',
               'npz_file': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm_PL/images/tip_vortex_pl/vortex_core_results.npz'},
    '2025-T': {'pl_master_path': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm-transition_PL',
               'npz_file': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm-transition_PL/images/tip_vortex_pl/vortex_core_results_2025T.npz'},
}


def load_tv_case(case_label, tv_cfg):

    '''
    Reloads one TV tracking run's own saved track_results (no re-tracking - see
    track_vortex_core_sweep()'s own `results_savepath`), then re-grids the 3 wake ages (early/
    mid/late, same row_at_wake_age() targets tip_vortex_manager.py itself uses: 10/90/160 deg)
    fresh from the raw scattered exports - cheap (3 angles, not the full ~18-angle tracking
    sweep) compared to the tracking itself, so not worth caching separately.

    Returns
    -------
    track_results, rho_values, r_tip, u_tip, profiles_by_age, row_late
    '''

    data = np.load(tv_cfg['npz_file'], allow_pickle=True)
    track_results = list(data['results'])
    rho_values = data['rho_values']

    pl_master_path = tv_cfg['pl_master_path']
    r_tip = tv_cfg.get('r_tip', R_TIP)
    u_tip = tv_cfg.get('rpm', RPM) * 2 * np.pi / 60 * r_tip

    row_mid = row_at_wake_age(track_results, 90)
    row_early = row_at_wake_age(track_results, 10)
    row_late = row_at_wake_age(track_results, 160)

    def _profiles(row):
        angle_str = f'{int(round(row["angle"])):02d}'
        x_file = f'{pl_master_path}/data/x-vel/vel_x_{angle_str}deg_avg.txt'
        y_file = f'{pl_master_path}/data/y-vel/vel_y_{angle_str}deg_avg.txt'
        z_file = f'{pl_master_path}/data/z-vel/vel_z_{angle_str}deg_avg.txt'
        R, Y, v_r, v_a, v_t = scattered_cylindrical_components(
            x_file, y_file, z_file, angle_deg=float(row['angle']), n_r=250, n_y=125,
        )
        v_swirl_mean, v_swirl_std, v_along_mean, v_along_std = azimuthal_profiles(
            R, Y, v_r, v_a, v_t, row['r_core'], row['y_core'], rho_values,
        )
        p_R, p_Y, p_grid = griddata_scattered_field(
            f'{pl_master_path}/data/pstat/p_stat_{angle_str}deg_avg.txt', float(row['angle']), n_r=250, n_y=125,
        )
        p_mean, p_std = azimuthal_scalar_profile(p_R, p_Y, p_grid, row['r_core'], row['y_core'], rho_values)
        return v_swirl_mean, v_swirl_std, v_along_mean, v_along_std, p_mean, p_std

    profiles_by_age = {'early': _profiles(row_early), 'mid': _profiles(row_mid), 'late': _profiles(row_late)}

    return track_results, rho_values, r_tip, u_tip, profiles_by_age, row_late


def main_tv_compare():

    os.makedirs(os.path.join(MASTER_PATH, 'images/comparison/tv'), exist_ok=True)

    track_results_by_case, profiles_by_age_by_case = {}, {}
    v_swirl_mean_by_case, row_late_by_case = {}, {}
    r_tip, u_tip, rho_values = None, None, None

    for case_label, tv_cfg in TV_CASES.items():
        print(40 * '-')
        print(f"Loading TV tracking results for case '{case_label}': {tv_cfg['npz_file']}")
        (track_results, rho_values, r_tip, u_tip_i,
         profiles_by_age, row_late) = load_tv_case(case_label, tv_cfg)
        track_results_by_case[case_label] = track_results
        profiles_by_age_by_case[case_label] = profiles_by_age
        v_swirl_mean_by_case[case_label] = profiles_by_age['late'][0]
        row_late_by_case[case_label] = row_late
        u_tip = u_tip_i  # assumed shared across cases (same rotor) - see plot_core_trajectory_compare()'s docstring

    print(40 * '-')
    print('Plotting cross-case core trajectory comparison')
    plot_core_trajectory_compare(
        track_results_by_case, r_tip,
        savepath=os.path.join(MASTER_PATH, 'images/comparison/tv/core_trajectory_compare.png'),
    )

    print(40 * '-')
    print('Plotting cross-case Fig. 6 (tangential/axial-deficit/pressure) comparison')
    plot_fig6_compare(
        rho_values, r_tip, u_tip, PREF, profiles_by_age_by_case,
        savepath_tangential=os.path.join(MASTER_PATH, 'images/comparison/tv/tangential_nd_compare.png'),
        savepath_axial_deficit=os.path.join(MASTER_PATH, 'images/comparison/tv/axial_deficit_nd_compare.png'),
        savepath_pressure=os.path.join(MASTER_PATH, 'images/comparison/tv/pressure_nd_compare.png'),
    )

    print(40 * '-')
    print('Plotting cross-case Fig. 7a tangential-models comparison')
    tv_models = {'lo': lamb_oseen, 'scully': scully, 'taylor': taylor}
    tv_model_labels = {'lo': 'Lamb-Oseen', 'scully': 'Scully', 'taylor': 'Taylor'}
    tv_model_colors = {'lo': 'tab:red', 'scully': 'tab:blue', 'taylor': 'tab:green'}
    plot_fig7_compare(
        rho_values, r_tip, u_tip, v_swirl_mean_by_case, row_late_by_case,
        tv_models, tv_model_labels, tv_model_colors, model_names=('lo', 'scully', 'taylor'),
        savepath=os.path.join(MASTER_PATH, 'images/comparison/tv/tangential_models_compare.png'),
    )


if __name__ == '__main__':
    # main()                  # cross-case StripForces comparison (CASES dict above)
    main_radii_compare()      # cross-case Cp/Cf-vs-x/c comparison (CASES dict above, now also
                              # needs each case's own inst_pressure_file) - run on a COMPUTE NODE
                              # (sbatch), not interactively: FrictionLines._load() eagerly loads
                              # the full per-frame force field (unlike SurfaceVariable, which
                              # streams) - confirmed to need ~100-200+ GB per case on this
                              # project's own real files, even after the azimuth_center_deg/
                              # azimuth_tol_deg crop (see AZIMUTH_CENTER_DEG's own comment).
    # main_tv_compare()       # cross-case TV tracking comparison (TV_CASES dict above) - comment
                              # this out and uncomment one of the others to run that one instead;
                              # only one of the three should run per invocation.
