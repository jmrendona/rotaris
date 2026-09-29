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
    '2025':     {'inst_force_file': '/path/to/2025/2025_forces_rotor.h5'},       # TODO
    '2025-T':   {'inst_force_file': '/path/to/2025-T/2025T_forces_rotor.h5'},    # TODO
    '2026':     {'inst_force_file': '/path/to/2026/2026_forces_rotor.h5'},       # TODO
    '2026-DNS': {'inst_force_file': '/path/to/2026-DNS/2026DNS_forces_rotor.h5'},  # TODO
}

# Shared rotor/geometry/crop parameters - see cases/6e-5_6000rpm_HF.yaml
# for what each of these means; override per-case below (e.g. CASES
# ['2026-DNS']['rpm'] = ...) only if one case genuinely differs.
R_TIP = 0.125
RPM = 6000
DT = 0.000056
SPAN_MIN = 0.02
SPAN_AXIS, CHORD_AXIS, THICKNESS_AXIS = 0, 2, 1
VALIDATE_AXES = True
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


def _resolve(path):
    '''Relative CASES paths are resolved against this file's own directory - same convention as manager.py.'''
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def load_case(case_label, case_cfg):

    '''
    Build one case's StripForces instance, its compute() result (for the
    raw time-trace comparison), and its phase_lock() result (for the
    phase-locked/polar comparison) - the two inputs plot_time_trace_compare()/
    plot_vs_angle_compare() need, one dict entry per case.
    '''

    print(40 * '-')
    print(f"Loading case '{case_label}': {case_cfg['inst_force_file']}")

    sf = StripForces(
        _resolve(case_cfg['inst_force_file']),
        r_tip=case_cfg.get('r_tip', R_TIP), rpm=case_cfg.get('rpm', RPM),
        span_min=case_cfg.get('span_min', SPAN_MIN),
        span_axis=case_cfg.get('span_axis', SPAN_AXIS), chord_axis=case_cfg.get('chord_axis', CHORD_AXIS),
        thickness_axis=case_cfg.get('thickness_axis', THICKNESS_AXIS),
        validate_axes=case_cfg.get('validate_axes', VALIDATE_AXES),
    )

    result = sf.compute(span_min=case_cfg.get('span_min', SPAN_MIN), n_span_bins=case_cfg.get('n_span_bins', N_SPAN_BINS))
    phase_locked = sf.phase_lock(result, dt=case_cfg.get('dt', DT), n_azimuth_bins=72)

    return result, phase_locked


def main():

    results = {}
    phase_locked_all = {}

    for case_label, case_cfg in CASES.items():
        result, phase_locked = load_case(case_label, case_cfg)
        results[case_label] = result
        phase_locked_all[case_label] = phase_locked

    print(40 * '-')
    print('Plotting cross-case raw time-trace comparison (Fig. 5.6 redesign)')
    plot_time_trace_compare(
        results, dt=DT, strips=STRIPS,
        savepath=os.path.join(MASTER_PATH, 'images/forces/comparison/force_time_trace_compare.pdf'),
    )

    print(40 * '-')
    print('Plotting cross-case phase-locked/polar comparison (Fig. 5.7 redesign)')
    plot_vs_angle_compare(
        phase_locked_all, strips=STRIPS,
        savepath=os.path.join(MASTER_PATH, 'images/forces/comparison/force_vs_angle_compare.pdf'),
    )


if __name__ == '__main__':
    main()
