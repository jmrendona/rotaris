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
from bladeprocessor.case_comparison import plot_time_trace_compare, plot_vs_angle_compare
from bladeprocessor.friction_lines import FrictionLines
from bladeprocessor.surface_variable import SurfaceVariable
from bladeprocessor.pod import pod
from bladeprocessor.dmd import joint_dmd, plot_joint_dmd_spectrum, plot_dmd_mode
from case_config import load_case_config

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
    '2025':     {'inst_force_file': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm/2025_forces_rotor.h5'},       # TODO
    '2025-T':   {'inst_force_file': '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm-transition/2025T_forces_rotor.h5'},    # TODO
    '2026':     {'inst_force_file': '/scratch/jmrendon/Rotor-alone/15e-6_6000rpm/2026_forces_rotor.h5'},       # TODO
    '2026-DNS': {'inst_force_file': '/scratch/jmrendon/Rotor-alone/1e-5_6000rpm-dns/2026DNS_forces_rotor.h5'},  # TODO
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


# ------------- Cross-QUANTITY (Cf vs Cp) joint DMD, ONE case ------------- #
#
# A DIFFERENT kind of comparison from the cross-SIMULATION StripForces
# one above: same case, same run, but combining Cf and Cp - two
# different physical quantities, recorded at different native sampling
# rates (dt_forces != dt_pressure) - into ONE shared DMD fit (see
# bladeprocessor.dmd.joint_dmd()'s own docstring for the full
# derivation, including why/how the finer-sampled field is resampled
# down onto the coarser field's own grid before fitting). This answers a
# genuinely different question than skin_friction_manager.py's/
# pressure_manager.py's own SEPARATE per-quantity dmd() calls: not "do
# Cf and Cp each show a persistent mode on their own", but "does a
# SINGLE, COMBINED linear operator find dynamics that genuinely mix
# both fields together" - see joint_dmd()'s own 'fraction_a'/
# 'fraction_b' for how to read that off the result.
#
# Uses this project's normal --case/cases/*.yaml loader
# (case_config.load_case_config()), NOT the CASES dict above (that dict
# is for the StripForces cross-case comparison only, and has its own,
# separate per-case config). Run with:
#   python manager_comparison.py --case cases/<your case>.yaml
# and call main_joint_dmd() instead of main() at the bottom (see the
# __main__ dispatch below) - both paths share this file's own argparse
# via load_case_config(), so only one can run per invocation.

# pod_span_min/pod_span_max/pod_chord_min/pod_chord_max/pod_point_stride,
# for BOTH Cf and Cp: copy the SAME values this case's own
# skin_friction_manager.py/pressure_manager.py runs already settled on
# (see those files' own identically-named variables) - this section
# deliberately does NOT re-derive a good region from scratch, since that
# work was already done there.
JOINT_DMD_CF_SPAN_MIN, JOINT_DMD_CF_SPAN_MAX = 0.1 * R_TIP, 1 * R_TIP  # TODO: copy from skin_friction_manager.py
JOINT_DMD_CF_CHORD_MIN, JOINT_DMD_CF_CHORD_MAX = None, None  # TODO
JOINT_DMD_CF_POINT_STRIDE = None  # TODO
JOINT_DMD_CP_SPAN_MIN, JOINT_DMD_CP_SPAN_MAX = None, None  # TODO: copy from pressure_manager.py
JOINT_DMD_CP_CHORD_MIN, JOINT_DMD_CP_CHORD_MAX = None, None  # TODO
JOINT_DMD_CP_POINT_STRIDE = 10  # TODO

JOINT_DMD_N_MODES_CF = 10  # TODO: match skin_friction_manager.py's own dmd() n_modes for this case
JOINT_DMD_N_MODES_CP = 10  # TODO: match pressure_manager.py's own dmd() n_modes for this case


def _crop_region(radius, chord, span_min, span_max, chord_min, chord_max, point_stride):

    '''Same span/chord/stride cropping logic as skin_friction_manager.py's/pressure_manager.py's own pod_region block, factored out since it's needed identically for both Cf and Cp here.'''

    region = (radius >= span_min) & (radius <= span_max)
    if chord_min is not None:
        region &= chord >= chord_min
    if chord_max is not None:
        region &= chord <= chord_max
    if point_stride is not None and point_stride > 1:
        kept_indices = np.flatnonzero(region)[::point_stride]
        region = np.zeros_like(region)
        region[kept_indices] = True
    return region


def load_joint_dmd_pod_results():

    '''
    Builds the Cf (magnitude, Upper surface) and Cp POD results for the
    CURRENT --case (see case_config.load_case_config()), restricted to
    the JOINT_DMD_CF_*/JOINT_DMD_CP_* region above - fill those in from
    this case's own skin_friction_manager.py/pressure_manager.py runs
    before calling this.

    Returns
    -------
    cfg, cf_pod_result, radius_fl_region, chord_fl_region, cp_pod_result, radius_sv_region, chord_sv_region
    '''

    cfg = load_case_config()

    print(40 * '-')
    print(f"Loading Cf for case '{cfg.case}': {cfg.inst_force_file}")
    fl = FrictionLines(
        os.path.join(cfg.master_path, cfg.inst_force_file),
        r_tip=cfg.r_tip, rho_ref=cfg.rho_ref, rpm=cfg.rpm, c_ref=cfg.c_ref,
        span_axis=cfg.span_axis, chord_axis=cfg.chord_axis, thickness_axis=cfg.thickness_axis,
        span_min=cfg.span_min, validate_axes=cfg.validate_axes,
    )
    radius_fl = fl._radius('Upper')
    _, chord_fl = fl._span_chord('Upper')
    area_fl = fl._area('Upper')
    cf_region = _crop_region(radius_fl, chord_fl, JOINT_DMD_CF_SPAN_MIN, JOINT_DMD_CF_SPAN_MAX,
                              JOINT_DMD_CF_CHORD_MIN, JOINT_DMD_CF_CHORD_MAX, JOINT_DMD_CF_POINT_STRIDE)
    cf_data = fl.cf_time_series(surface='Upper', component=None)[:, cf_region]
    cf_pod_result = pod(cf_data, n_modes=JOINT_DMD_N_MODES_CF, weights=area_fl[cf_region])

    print(40 * '-')
    print(f"Loading Cp for case '{cfg.case}': {cfg.inst_pressure_file}")
    sv = SurfaceVariable(
        os.path.join(cfg.master_path, cfg.inst_pressure_file),
        r_tip=cfg.r_tip, rho_ref=cfg.rho_ref, rpm=cfg.rpm, pref=cfg.pref, c_ref=cfg.c_ref,
        span_axis=cfg.span_axis, chord_axis=cfg.chord_axis, thickness_axis=cfg.thickness_axis,
    )
    radius_sv = sv._radius('Upper')
    _, chord_sv = sv._span_chord('Upper')
    cp_region = _crop_region(radius_sv, chord_sv, JOINT_DMD_CP_SPAN_MIN, JOINT_DMD_CP_SPAN_MAX,
                              JOINT_DMD_CP_CHORD_MIN, JOINT_DMD_CP_CHORD_MAX, JOINT_DMD_CP_POINT_STRIDE)
    cp_weights = sv._area('Upper')[cp_region] if sv.areas['Upper'] is not None else None
    if cfg.normalize:
        cp_data = sv.cp_time_series_region(surface='Upper', region_mask=cp_region)[:-2]
    else:
        cp_data = sv.variable_time_series_region('static_pressure', surface='Upper', region_mask=cp_region)[:-2]
    cp_pod_result = pod(cp_data, n_modes=JOINT_DMD_N_MODES_CP, weights=cp_weights)

    return (cfg, cf_pod_result, radius_fl[cf_region], chord_fl[cf_region],
            cp_pod_result, radius_sv[cp_region], chord_sv[cp_region])


def main_joint_dmd():

    os.makedirs(os.path.join(MASTER_PATH, 'images/joint_dmd'), exist_ok=True)

    (cfg, cf_pod_result, radius_fl_region, chord_fl_region,
     cp_pod_result, radius_sv_region, chord_sv_region) = load_joint_dmd_pod_results()

    print(40 * '-')
    print(f"Running joint DMD (Cf + Cp) for case '{cfg.case}'")
    joint_result = joint_dmd(cf_pod_result, cfg.dt_forces, cp_pod_result, cfg.dt_pressure,
                              n_modes_a=JOINT_DMD_N_MODES_CF, n_modes_b=JOINT_DMD_N_MODES_CP)
    print(f"Resampled field: {joint_result['resampled_field']} (down onto the other's coarser grid, "
          f"dt_common={joint_result['dt']:.4g} s)")
    print('Joint frequencies [Hz]:   ', joint_result['frequency'])
    print('Joint growth rates [1/s]: ', joint_result['growth_rate'])
    print('fraction_a (Cf share):    ', joint_result['fraction_a'])
    print('fraction_b (Cp share):    ', joint_result['fraction_b'])

    print(40 * '-')
    print('Plotting joint DMD spectrum (color = fraction in Cf, 0.5 = genuinely shared)')
    plot_joint_dmd_spectrum(
        joint_result,
        savepath=os.path.join(MASTER_PATH, f"images/joint_dmd/joint_dmd_spectrum_{cfg.case}.png"),
    )

    # The mode closest to 0.5/0.5 is the one actually worth a closer,
    # spatial look - the most genuinely SHARED dynamics the joint fit
    # found, as opposed to one field's own content incidentally present
    # in the combined fit (see joint_dmd()'s own docstring).
    most_shared = int(np.argmin(np.abs(joint_result['fraction_a'] - 0.5)))
    print(f"Mode {most_shared + 1} is the most evenly shared (fraction_a={joint_result['fraction_a'][most_shared]:.3f}) "
          f"- plotting its spatial pattern in both fields")

    print(40 * '-')
    print('Plotting the most-shared joint DMD mode - Cf side')
    plot_dmd_mode(
        radius_fl_region, chord_fl_region,
        {'modes': joint_result['modes_a'], 'frequency': joint_result['frequency'],
         'growth_rate': joint_result['growth_rate']},
        mode_index=most_shared, unit_label='$C_f$ joint-DMD mode amplitude [-]',
        savepath=os.path.join(MASTER_PATH, f"images/joint_dmd/joint_dmd_mode_cf_{cfg.case}.png"),
    )

    print(40 * '-')
    print('Plotting the most-shared joint DMD mode - Cp side')
    plot_dmd_mode(
        radius_sv_region, chord_sv_region,
        {'modes': joint_result['modes_b'], 'frequency': joint_result['frequency'],
         'growth_rate': joint_result['growth_rate']},
        mode_index=most_shared, unit_label='Cp joint-DMD mode amplitude [-]',
        savepath=os.path.join(MASTER_PATH, f"images/joint_dmd/joint_dmd_mode_cp_{cfg.case}.png"),
    )


if __name__ == '__main__':
    main()                  # cross-case StripForces comparison (CASES dict above)
    # main_joint_dmd()      # cross-quantity Cf/Cp joint DMD for ONE case - needs --case
                            # (fill in JOINT_DMD_CF_*/JOINT_DMD_CP_* above first); comment
                            # out main() above and uncomment this to run it instead - both
                            # share this file's own argparse (via load_case_config()), so
                            # only one of the two should run per invocation.
