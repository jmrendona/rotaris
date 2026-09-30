'''
Shared --case argument parsing + cases/*.yaml loading for every
skin_friction_manager.py / pressure_manager.py / forces_manager.py /
tip_vortex_manager.py script - one place to add/rename a YAML key
instead of four. See cases/*.yaml's own header comment for what each
field means, and any manager script for how it's used downstream.
'''

import os
import argparse
from types import SimpleNamespace

import yaml


def load_case_config():

    '''
    Parse --case (default cases/6e-5_6000rpm_HF.yaml, resolved relative
    to THIS file's own directory unless given as an absolute path - so
    an existing submit script that runs a manager with no --case at all
    keeps using that default unchanged), load its YAML, and return every
    field as an attribute on a plain namespace - e.g. cfg.master_path,
    cfg.r_tip, cfg.span_axis. Each manager script unpacks only the
    fields it actually uses into local names, right after calling this
    (same bare names as before this file existed - master_path, r_tip,
    etc. - so the rest of each manager's body reads exactly as it did
    when this was all one file).

    Returns
    -------
    types.SimpleNamespace
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='cases/6e-5_6000rpm_HF.yaml',
                         help='Path to the case config YAML (see cases/*.yaml) - relative to '
                              'this file\'s own directory unless given as an absolute path.')
    args = parser.parse_args()

    case_config_path = args.case
    if not os.path.isabs(case_config_path):
        case_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_config_path)

    with open(case_config_path) as f:
        cfg = yaml.safe_load(f)

    return SimpleNamespace(
        master_path=cfg['master_path'],
        case=cfg['case'],

        inst_force_file=cfg['files']['inst_force'],
        avg_force_file=cfg['files']['avg_force'],
        inst_pressure_file=cfg['files']['inst_pressure'],
        avg_pressure_file=cfg['files']['avg_pressure'],

        r_tip=cfg['rotor']['r_tip'],
        rho_ref=cfg['rotor']['rho_ref'],
        rpm=cfg['rotor']['rpm'],
        pref=cfg['rotor']['pref'],

        span_axis=cfg['axes']['span'],
        chord_axis=cfg['axes']['chord'],
        thickness_axis=cfg['axes']['thickness'],
        validate_axes=not cfg['axes']['skip_validation'],

        span_min=cfg['crop']['span_min'],

        reverse_chord=cfg['friction']['reverse_chord'],
        radii=cfg['friction']['radii'],

        dt=cfg['convergence']['dt'],

        blade_figsize=tuple(cfg['blade_figsize']) if cfg['blade_figsize'] else None,

        frame_loop_step=cfg['frame_loop_step'],
    )
