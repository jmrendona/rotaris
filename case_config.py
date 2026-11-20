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

        # All four are now optional (.get(), not bare ['...']) - a case
        # with only a forces conversion done so far (e.g. a newly-added
        # stator/beam case with no pressure or PowerFLOW-side average
        # export yet) shouldn't have to invent placeholder filenames for
        # the other three just to load; skin_friction_manager.py/
        # forces_manager.py/pressure_manager.py only reference avg_force/
        # avg_pressure in their own commented-out "average" sections
        # today anyway. Whichever file a manager actually opens still
        # fails loudly at that point (os.path.join/h5py.File on a missing
        # or None path) if it was genuinely needed but never set.
        inst_force_file=cfg['files'].get('inst_force'),
        avg_force_file=cfg['files'].get('avg_force'),
        inst_pressure_file=cfg['files'].get('inst_pressure'),
        avg_pressure_file=cfg['files'].get('avg_pressure'),

        # r_tip/rpm are now optional (.get(), not bare ['...']) - meaningful
        # for a rotor (and r_tip for a stator, which has a real annulus
        # radius too) but not for a beam/strut, which has no rotation rate
        # and no outer "tip radius" of its own - see object_type/v_ref
        # below. A case yaml that omits them (any pre-existing rotor case
        # with rpm/r_tip set keeps working exactly as before) gets None,
        # same as c_ref already did.
        r_tip=cfg['rotor'].get('r_tip'),
        rho_ref=cfg['rotor']['rho_ref'],
        rpm=cfg['rotor'].get('rpm'),
        pref=cfg['rotor']['pref'],
        c_ref=cfg['rotor'].get('c_ref'),

        # Reference freestream/inlet velocity [m/s] - the Cf/Cp dynamic-
        # pressure reference for a STATIONARY object (q_ref =
        # 0.5*rho_ref*v_ref^2, uniform - see
        # bladeprocessor._dynamics.dynamic_pressure), in place of rpm's
        # rotor formula (q_ref local to each point's radius). Required
        # (instead of rpm) for Cf/Cp on a stator/beam/casing case - None
        # (default) for a rotor case, which uses rpm instead.
        v_ref=cfg['rotor'].get('v_ref'),

        # 'rotor' (default), 'stator', 'beam', or 'casing' - which kind of
        # part this case's .snc/HDF5 data is for (see convert.py's
        # --object and the README's extension-to-stationary-objects
        # notes). Informational/dispatch only here - managers use it to
        # decide e.g. whether to pass rpm or v_ref into FrictionLines/
        # SurfaceVariable, and whether radii means physical radius
        # (rotor/stator) or span-axis stations (beam).
        object_type=cfg.get('object_type', 'rotor'),

        # Which surface skin_friction_manager.py/pressure_manager.py
        # operate on - 'Upper' (default, matches every case before this
        # existed) or 'Lower'. A per-case setting rather than editing
        # every single manager call by hand: switching sides is a common,
        # recurring thing to want to do quickly, and every output
        # filename already folds in surface.lower() (see
        # skin_friction_manager.py) so an Upper run and a Lower run never
        # silently overwrite each other's images.
        surface=cfg.get('surface', 'Upper'),

        # Whether friction_lines()/plot_variable_surface() grid the
        # scattered field (scipy.interpolate.griddata) before rendering,
        # instead of a raw scatter - see either method's own interpolate
        # docstring. A per-case setting, not a hardcoded call-site
        # argument: whether this is worth turning on depends on this
        # case's OWN surfel density (on, confirmed useful, for a stator
        # vane's/beam's much sparser mesh; off, the original default, for
        # a full rotor blade's much denser one) - putting it here means a
        # manager script doesn't need every single friction_lines() call
        # edited by hand, and whoever copies this case's yaml for a
        # similar one can't forget to set it on one call but not another.
        interpolate=cfg.get('interpolate', False),

        normalize=cfg.get('normalize', False),

        span_axis=cfg['axes']['span'],
        chord_axis=cfg['axes']['chord'],
        thickness_axis=cfg['axes']['thickness'],
        validate_axes=not cfg['axes']['skip_validation'],

        span_min=cfg['crop']['span_min'],
        # Optional (.get()) - None (default) means no upper crop, matching
        # every case before this existed. Needed on a case whose mesh has
        # something ELSE past the part's real tip sharing the same
        # azimuth/face (e.g. a shroud band right past a stator vane's own
        # tip, confirmed on a real case: without span_max, a radius/chord
        # map showed a gap then a disconnected island of points past the
        # true tip, and cf_at_radii() near there mixed in shroud points).
        span_max=cfg['crop'].get('span_max'),

        # Both None (default, matches every case before these existed):
        # no azimuth crop. Set both on a case whose .snc mesh combines
        # multiple blades into one face (no per-blade face_name to
        # select at conversion time - confirmed for this project's own
        # Rotor-strut case) - see bladeprocessor._azimuth.azimuth_deg
        # and FrictionLines/StripForces/SurfaceVariable's own
        # azimuth_center_deg/azimuth_tol_deg docstrings for the full
        # mechanism. Optional key - cfg['crop'].get(...), not cfg['crop'][...],
        # so a case yaml written before this existed still loads unchanged.
        azimuth_center_deg=cfg['crop'].get('azimuth_center_deg'),
        azimuth_tol_deg=cfg['crop'].get('azimuth_tol_deg'),

        reverse_chord=cfg['friction']['reverse_chord'],
        radii=cfg['friction']['radii'],

        # Separate, NOT interchangeable: the forces and pressure branches
        # (different raw exports - converters/snc_reader.py vs
        # converters/ensight_to_h5.py) can and do have different
        # physical frame intervals for the same case - confirmed a real
        # case where reusing one shared `dt` for both would have
        # mis-scaled one branch's frequency axis by an order of
        # magnitude. skin_friction_manager.py/forces_manager.py use
        # dt_forces; pressure_manager.py uses dt_pressure.
        dt_forces=cfg['convergence']['dt_forces'],
        dt_pressure=cfg['convergence']['dt_pressure'],

        blade_figsize=tuple(cfg['blade_figsize']) if cfg['blade_figsize'] else None,

        frame_loop_step=cfg['frame_loop_step'],
    )
