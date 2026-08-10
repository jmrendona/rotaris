import argparse
import os

import numpy as np

from converters.snc_reader import SNCReader
from converters.ensight_to_h5 import convert_snc_to_h5
from converters import fnc_plane

'''
CLI entry point for the .snc -> HDF5 and .fnc -> HDF5 conversion branches
(see README.md):

    convert.py forces        ...   SNCReader.to_h5 - forces, skin friction,
                                    normals, geometry, straight from the raw
                                    .snc. No pf2ens needed.
    convert.py pressure      ...   convert_snc_to_h5 - Static Pressure via
                                    pf2ens (the only trusted source for
                                    pressure). Calls pf2ens once per frame
                                    internally.
    convert.py fnc-meridional...   fnc_plane.extract_to_h5 on ONE rotated
                                    meridional (hub-to-tip, inlet-to-outlet)
                                    plane - the "radial-cuts" shape used for
                                    vortex tracking.
    convert.py fnc-meridional-sweep    same, but MULTIPLE angles (a start/
                                    end/step range) in one command, one HDF5
                                    file per angle, sharing a single
                                    nc-stats.ri call across all of them.
    convert.py fnc-iso-radius...   fnc_plane.extract_to_h5 on a fixed-radius
                                    cylindrical surface (e.g. 75% span).
    convert.py fnc-points     ...  fnc_plane.extract_to_h5 on an arbitrary
                                    user-supplied point cloud (plain text,
                                    one "x y z" per line).
    convert.py fnc-freeze-mask...  (re)compute Data/frozen on an already-
                                    extracted fnc-* HDF5 file, in place - no
                                    re-extraction. See fnc_plane.add_frozen_mask
                                    and extract_to_h5's freeze_mask_variable -
                                    vtkValidPointMask alone can miss body-
                                    fixed solid geometry (confirmed on a
                                    stator hub); this flags points whose
                                    value barely changes across frames, the
                                    signature of a lattice cell PowerFLOW
                                    never updates because it's inside solid
                                    geometry the .fnc measurement volume
                                    didn't exclude.
    convert.py fnc-plot       ...  quick-look contour plot from an already-
                                    extracted fnc-meridional/fnc-iso-radius
                                    HDF5 file - no re-extraction, just
                                    fnc_plane.plot_frame. Automatically
                                    combines Data/valid with Data/frozen,
                                    if present. NOTE: there is no working
                                    data mask for rotor blades - only a
                                    validated visual annotation
                                    (fnc_plane.RotorBladePosition.
                                    blade_azimuths_deg/blade_extents_deg,
                                    Python API only, not wired into this
                                    CLI) - see that class's docstring.

Meant to be run as a cluster batch job - see run_conversion.sh. The fnc-*
subcommands additionally need `pf2ens` on $PATH (PowerFLOW 6-2024-R1 -
source /project/rrg-moreaust-ac/Env/powerflow_env.sh 6-2024-R1) and a
Python environment with pyvista/vtk importable (see fnc_plane.py's module
docstring for the working module-load recipe).
'''


def run_forces(args):
    reader = SNCReader(args.snc_path)
    reader.to_h5(args.output, face_name=args.face_name, surface_split=args.surface_split)
    reader.close()
    print(f'wrote {args.output}')


def run_pressure(args):
    convert_snc_to_h5(
        args.snc_path,
        args.output,
        args.first,
        args.last,
        nc_stats_path=args.nc_stats,
        reference_frame=args.reference_frame,
        work_dir=args.work_dir,
        surface_split=args.surface_split,
    )
    print(f'wrote {args.output}')


def _resolve_nc_stats(fnc_path, nc_stats_path, work_dir):
    if nc_stats_path is not None:
        return nc_stats_path
    import tempfile
    stats_dir = work_dir or tempfile.mkdtemp(prefix='fnc_nc_stats_')
    os.makedirs(stats_dir, exist_ok=True)
    return fnc_plane.run_nc_stats(fnc_path, os.path.join(stats_dir, 'nc_stats.txt'))


def _maybe_plot(args, variables):
    if not args.plot:
        return
    plot_variable = args.plot_variable or variables[0]
    plot_frame_index = args.plot_frame if args.plot_frame is not None else args.first
    fnc_plane.plot_frame(args.output, plot_variable, frame_index=plot_frame_index, savepath=args.plot)
    print(f'wrote {args.plot}')


def run_fnc_meridional(args):
    variables = args.variables.split(',')
    nc_stats_path = _resolve_nc_stats(args.fnc_path, args.nc_stats, args.work_dir)

    bbox = None
    if args.inplane_range is None or args.axial_range is None:
        bbox = fnc_plane.parse_bounding_box_mks(nc_stats_path)

    points, grid_shape = fnc_plane.meridional_plane_points(
        angle_deg=args.angle,
        axis=args.axis,
        n_inplane=args.n_inplane,
        n_axial=args.n_axial,
        inplane_range=tuple(args.inplane_range) if args.inplane_range else None,
        axial_range=tuple(args.axial_range) if args.axial_range else None,
        bbox=bbox,
    )

    fnc_plane.extract_to_h5(
        args.fnc_path, points, variables, args.output,
        frames=list(range(args.first, args.last + 1)),
        grid_shape=grid_shape,
        geometry_attrs={'kind': 'meridional', 'angle_deg': args.angle, 'axis': args.axis},
        freeze_mask_variable=args.freeze_mask_variable, freeze_rel_threshold=args.freeze_rel_threshold,
        work_dir=args.work_dir, nc_stats_path=nc_stats_path,
    )
    print(f'wrote {args.output}')
    _maybe_plot(args, variables)


def run_fnc_meridional_sweep(args):
    variables = args.variables.split(',')
    nc_stats_path = _resolve_nc_stats(args.fnc_path, args.nc_stats, args.work_dir)

    bbox = None
    if args.inplane_range is None or args.axial_range is None:
        bbox = fnc_plane.parse_bounding_box_mks(nc_stats_path)

    os.makedirs(args.output_dir, exist_ok=True)

    n_steps = round((args.angle_end - args.angle_start) / args.angle_step)
    angles = [args.angle_start + i * args.angle_step for i in range(n_steps + 1)]
    frames = list(range(args.first, args.last + 1))

    print(f'Sweeping {len(angles)} angles ({angles[0]:g} to {angles[-1]:g} deg, step {args.angle_step:g}), '
          f'{len(frames)} frames each - {len(angles) * len(frames)} total pf2ens calls.')

    for angle in angles:
        output_path = os.path.join(args.output_dir, f'{args.prefix}_{angle:g}deg.h5')
        print(f'=== angle {angle:g} deg -> {output_path} ===')

        points, grid_shape = fnc_plane.meridional_plane_points(
            angle_deg=angle,
            axis=args.axis,
            n_inplane=args.n_inplane,
            n_axial=args.n_axial,
            inplane_range=tuple(args.inplane_range) if args.inplane_range else None,
            axial_range=tuple(args.axial_range) if args.axial_range else None,
            bbox=bbox,
        )

        fnc_plane.extract_to_h5(
            args.fnc_path, points, variables, output_path,
            frames=frames,
            grid_shape=grid_shape,
            geometry_attrs={'kind': 'meridional', 'angle_deg': angle, 'axis': args.axis},
            freeze_mask_variable=args.freeze_mask_variable, freeze_rel_threshold=args.freeze_rel_threshold,
            work_dir=args.work_dir, nc_stats_path=nc_stats_path,
        )
        print(f'wrote {output_path}')

    print(f'Sweep done: {len(angles)} files written to {args.output_dir}')


def run_fnc_iso_radius(args):
    variables = args.variables.split(',')
    nc_stats_path = _resolve_nc_stats(args.fnc_path, args.nc_stats, args.work_dir)

    bbox = None
    if args.axial_range is None:
        bbox = fnc_plane.parse_bounding_box_mks(nc_stats_path)

    points, grid_shape = fnc_plane.iso_radius_points(
        radius=args.radius,
        axis=args.axis,
        n_theta=args.n_theta,
        n_axial=args.n_axial,
        theta_range=tuple(args.theta_range) if args.theta_range else (0.0, 360.0),
        axial_range=tuple(args.axial_range) if args.axial_range else None,
        bbox=bbox,
    )

    fnc_plane.extract_to_h5(
        args.fnc_path, points, variables, args.output,
        frames=list(range(args.first, args.last + 1)),
        grid_shape=grid_shape,
        geometry_attrs={'kind': 'iso_radius', 'radius': args.radius, 'axis': args.axis},
        freeze_mask_variable=args.freeze_mask_variable, freeze_rel_threshold=args.freeze_rel_threshold,
        work_dir=args.work_dir, nc_stats_path=nc_stats_path,
    )
    print(f'wrote {args.output}')


def run_fnc_points(args):
    variables = args.variables.split(',')
    nc_stats_path = _resolve_nc_stats(args.fnc_path, args.nc_stats, args.work_dir)
    points = np.loadtxt(args.points_file, ndmin=2)

    fnc_plane.extract_to_h5(
        args.fnc_path, points, variables, args.output,
        frames=list(range(args.first, args.last + 1)),
        geometry_attrs={'kind': 'points'},
        freeze_mask_variable=args.freeze_mask_variable, freeze_rel_threshold=args.freeze_rel_threshold,
        work_dir=args.work_dir, nc_stats_path=nc_stats_path,
    )
    print(f'wrote {args.output}')


def run_fnc_plot(args):
    fnc_plane.plot_frame(args.h5_path, args.variable, frame_index=args.frame, savepath=args.output)
    print(f'wrote {args.output}')


def run_fnc_freeze_mask(args):
    fnc_plane.add_frozen_mask(args.h5_path, args.variable, rel_threshold=args.rel_threshold)


def build_parser():

    parser = argparse.ArgumentParser(
        description='rotaris: convert a PowerFLOW .snc surface measurement file to HDF5.'
    )
    subparsers = parser.add_subparsers(dest='mode', required=True)

    forces = subparsers.add_parser(
        'forces',
        help='Forces, Skin Friction, Normals, Geometry - straight from the raw .snc (SNCReader).',
    )
    forces.add_argument('snc_path', help='Path to the PowerFLOW surface measurement file (.snc)')
    forces.add_argument('output', help='Path to the output HDF5 file')
    forces.add_argument('--face-name', default=None,
                         help='Restrict to surfels matching this face (e.g. one rotor blade)')
    forces.add_argument('--surface-split', action='store_true',
                         help='Split into Upper/Lower surface groups (see SNCReader.surface_split)')
    forces.set_defaults(func=run_forces)

    pressure = subparsers.add_parser(
        'pressure',
        help='Static Pressure - via pf2ens (ensight_to_h5.convert_snc_to_h5).',
    )
    pressure.add_argument('snc_path', help='Path to the PowerFLOW surface measurement file (.snc)')
    pressure.add_argument('output', help='Path to the combined HDF5 output file')
    pressure.add_argument('--first', type=int, required=True, help='First frame to convert')
    pressure.add_argument('--last', type=int, required=True, help='Last frame to convert (inclusive)')
    pressure.add_argument('--nc-stats', default=None,
                           help='Path to saved `exaritool nc-stats.ri -detail` output')
    pressure.add_argument('--reference-frame', type=int, default=None,
                           help='Frame whose geometry is stored (default: --first)')
    pressure.add_argument('--work-dir', default=None,
                           help='Directory for intermediate pf2ens output (default: auto temp dir)')
    pressure.add_argument('--surface-split', action='store_true',
                           help='Split into Upper/Lower surface groups, classification borrowed '
                                'from the raw .snc file via nearest-neighbor matching (see '
                                'EnsightFrame.surface_split).')
    pressure.set_defaults(func=run_pressure)

    fnc_meridional = subparsers.add_parser(
        'fnc-meridional',
        help='Rotated meridional (hub-to-tip, inlet-to-outlet) plane from a .fnc fluid file '
             '(pf2ens + pyvista sampling, no PowerVIZ - see fnc_plane.py).',
    )
    fnc_meridional.add_argument('fnc_path', help='Path to the PowerFLOW fluid measurement file (.fnc)')
    fnc_meridional.add_argument('output', help='Path to the output HDF5 file')
    fnc_meridional.add_argument('--angle', type=float, required=True, help='Azimuthal angle, degrees')
    fnc_meridional.add_argument('--axis', default='z', choices=['x', 'y', 'z'],
                                 help="Rotor rotation axis (default: 'z')")
    fnc_meridional.add_argument('--n-inplane', type=int, default=100,
                                 help='Grid points across the in-plane (hub-to-tip) direction (default: 100)')
    fnc_meridional.add_argument('--n-axial', type=int, default=100,
                                 help='Grid points along the axial (inlet-to-outlet) direction (default: 100)')
    fnc_meridional.add_argument('--inplane-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                                 help='Signed in-plane extent, meters (default: file bounding box, from nc-stats.ri)')
    fnc_meridional.add_argument('--axial-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                                 help='Axial extent, meters (default: file bounding box, from nc-stats.ri)')
    fnc_meridional.add_argument('--variables', required=True,
                                 help="Comma-separated pf2ens variable codes (e.g. 'vmag,p') - "
                                      "run `pf2ens -d <fnc_path>` to list them for this file")
    fnc_meridional.add_argument('--first', type=int, required=True, help='First frame to extract')
    fnc_meridional.add_argument('--last', type=int, required=True, help='Last frame to extract (inclusive)')
    fnc_meridional.add_argument('--freeze-mask-variable', default=None,
                                 help='If given (needs --first/--last to span >= 2 frames), compute Data/frozen '
                                      'from this variable - flags points that barely change across frames, the '
                                      'signature of solid geometry vtkValidPointMask missed (confirmed on a '
                                      "body-fixed stator hub) - e.g. 'vmag'")
    fnc_meridional.add_argument('--freeze-rel-threshold', type=float, default=0.01,
                                 help='Fraction of the observed dynamic range below which a point counts as '
                                      'frozen (default: 0.01); only used with --freeze-mask-variable')
    fnc_meridional.add_argument('--work-dir', default=None,
                                 help='Directory for intermediate pf2ens output (default: auto temp dir)')
    fnc_meridional.add_argument('--nc-stats', default=None,
                                 help='Path to already-saved `exaritool nc-stats.ri -detail` output '
                                      '(default: run fresh)')
    fnc_meridional.add_argument('--plot', default=None,
                                 help='Also save a quick-look contour plot (PNG) after extraction')
    fnc_meridional.add_argument('--plot-variable', default=None,
                                 help='Which variable to plot (default: first in --variables)')
    fnc_meridional.add_argument('--plot-frame', type=int, default=None,
                                 help='Which frame index to plot (default: --first)')
    fnc_meridional.set_defaults(func=run_fnc_meridional)

    fnc_meridional_sweep = subparsers.add_parser(
        'fnc-meridional-sweep',
        help='Multiple rotated meridional planes in one command - one HDF5 file per angle, all sharing a '
             'single nc-stats.ri call (see fnc_plane.py).',
    )
    fnc_meridional_sweep.add_argument('fnc_path', help='Path to the PowerFLOW fluid measurement file (.fnc)')
    fnc_meridional_sweep.add_argument('output_dir', help='Directory to write one HDF5 file per angle into')
    fnc_meridional_sweep.add_argument('--prefix', default='plane',
                                       help="Output filename prefix - files are named '<prefix>_<angle>deg.h5' "
                                            "(default: 'plane')")
    fnc_meridional_sweep.add_argument('--angle-start', type=float, required=True, help='First angle, degrees')
    fnc_meridional_sweep.add_argument('--angle-end', type=float, required=True,
                                       help='Last angle, degrees (inclusive)')
    fnc_meridional_sweep.add_argument('--angle-step', type=float, required=True, help='Angle step, degrees')
    fnc_meridional_sweep.add_argument('--axis', default='z', choices=['x', 'y', 'z'],
                                       help="Rotor rotation axis (default: 'z')")
    fnc_meridional_sweep.add_argument('--n-inplane', type=int, default=100,
                                       help='Grid points across the in-plane (hub-to-tip) direction (default: 100)')
    fnc_meridional_sweep.add_argument('--n-axial', type=int, default=100,
                                       help='Grid points along the axial (inlet-to-outlet) direction (default: 100)')
    fnc_meridional_sweep.add_argument('--inplane-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                                       help='Signed in-plane extent, meters (default: file bounding box)')
    fnc_meridional_sweep.add_argument('--axial-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                                       help='Axial extent, meters (default: file bounding box)')
    fnc_meridional_sweep.add_argument('--variables', required=True,
                                       help="Comma-separated pf2ens variable codes (e.g. 'vmag,p') - "
                                            "run `pf2ens -d <fnc_path>` to list them for this file")
    fnc_meridional_sweep.add_argument('--first', type=int, required=True, help='First frame to extract (each angle)')
    fnc_meridional_sweep.add_argument('--last', type=int, required=True,
                                       help='Last frame to extract, inclusive (each angle)')
    fnc_meridional_sweep.add_argument('--freeze-mask-variable', default=None,
                                       help='If given (needs --first/--last to span >= 2 frames), compute '
                                            'Data/frozen from this variable for every angle - e.g. \'vmag\'')
    fnc_meridional_sweep.add_argument('--freeze-rel-threshold', type=float, default=0.01,
                                       help='Fraction of the observed dynamic range below which a point counts '
                                            'as frozen (default: 0.01); only used with --freeze-mask-variable')
    fnc_meridional_sweep.add_argument('--work-dir', default=None,
                                       help='Directory for intermediate pf2ens output (default: auto temp dir '
                                            'per angle)')
    fnc_meridional_sweep.add_argument('--nc-stats', default=None,
                                       help='Path to already-saved `exaritool nc-stats.ri -detail` output '
                                            '(default: run fresh, once, shared across all angles)')
    fnc_meridional_sweep.set_defaults(func=run_fnc_meridional_sweep)

    fnc_iso_radius = subparsers.add_parser(
        'fnc-iso-radius',
        help='Fixed-radius cylindrical surface (e.g. 75%% span) from a .fnc fluid file '
             '(pf2ens + pyvista sampling, no PowerVIZ - see fnc_plane.py).',
    )
    fnc_iso_radius.add_argument('fnc_path', help='Path to the PowerFLOW fluid measurement file (.fnc)')
    fnc_iso_radius.add_argument('output', help='Path to the output HDF5 file')
    fnc_iso_radius.add_argument('--radius', type=float, required=True, help='Physical radius, meters')
    fnc_iso_radius.add_argument('--axis', default='z', choices=['x', 'y', 'z'],
                                 help="Rotor rotation axis (default: 'z')")
    fnc_iso_radius.add_argument('--n-theta', type=int, default=360, help='Azimuthal grid resolution (default: 360)')
    fnc_iso_radius.add_argument('--n-axial', type=int, default=100, help='Axial grid resolution (default: 100)')
    fnc_iso_radius.add_argument('--theta-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                                 help='Azimuthal range, degrees (default: 0 360)')
    fnc_iso_radius.add_argument('--axial-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                                 help='Axial extent, meters (default: file bounding box, from nc-stats.ri)')
    fnc_iso_radius.add_argument('--variables', required=True,
                                 help="Comma-separated pf2ens variable codes (e.g. 'vmag,p') - "
                                      "run `pf2ens -d <fnc_path>` to list them for this file")
    fnc_iso_radius.add_argument('--first', type=int, required=True, help='First frame to extract')
    fnc_iso_radius.add_argument('--last', type=int, required=True, help='Last frame to extract (inclusive)')
    fnc_iso_radius.add_argument('--freeze-mask-variable', default=None,
                                 help='If given (needs --first/--last to span >= 2 frames), compute Data/frozen '
                                      'from this variable - flags points that barely change across frames, the '
                                      'signature of solid geometry vtkValidPointMask missed - e.g. \'vmag\'')
    fnc_iso_radius.add_argument('--freeze-rel-threshold', type=float, default=0.01,
                                 help='Fraction of the observed dynamic range below which a point counts as '
                                      'frozen (default: 0.01); only used with --freeze-mask-variable')
    fnc_iso_radius.add_argument('--work-dir', default=None,
                                 help='Directory for intermediate pf2ens output (default: auto temp dir)')
    fnc_iso_radius.add_argument('--nc-stats', default=None,
                                 help='Path to already-saved `exaritool nc-stats.ri -detail` output '
                                      '(default: run fresh)')
    fnc_iso_radius.set_defaults(func=run_fnc_iso_radius)

    fnc_points = subparsers.add_parser(
        'fnc-points',
        help='Arbitrary user-supplied point cloud from a .fnc fluid file '
             '(pf2ens + pyvista sampling, no PowerVIZ - see fnc_plane.py).',
    )
    fnc_points.add_argument('fnc_path', help='Path to the PowerFLOW fluid measurement file (.fnc)')
    fnc_points.add_argument('output', help='Path to the output HDF5 file')
    fnc_points.add_argument('--points-file', required=True,
                             help='Plain text file, one "x y z" row per query point (meters)')
    fnc_points.add_argument('--variables', required=True,
                             help="Comma-separated pf2ens variable codes (e.g. 'vmag,p') - "
                                  "run `pf2ens -d <fnc_path>` to list them for this file")
    fnc_points.add_argument('--first', type=int, required=True, help='First frame to extract')
    fnc_points.add_argument('--last', type=int, required=True, help='Last frame to extract (inclusive)')
    fnc_points.add_argument('--freeze-mask-variable', default=None,
                             help='If given (needs --first/--last to span >= 2 frames), compute Data/frozen '
                                  'from this variable - flags points that barely change across frames, the '
                                  'signature of solid geometry vtkValidPointMask missed - e.g. \'vmag\'')
    fnc_points.add_argument('--freeze-rel-threshold', type=float, default=0.01,
                             help='Fraction of the observed dynamic range below which a point counts as '
                                  'frozen (default: 0.01); only used with --freeze-mask-variable')
    fnc_points.add_argument('--work-dir', default=None,
                             help='Directory for intermediate pf2ens output (default: auto temp dir)')
    fnc_points.add_argument('--nc-stats', default=None,
                             help='Path to already-saved `exaritool nc-stats.ri -detail` output '
                                  '(default: run fresh)')
    fnc_points.set_defaults(func=run_fnc_points)

    fnc_freeze_mask = subparsers.add_parser(
        'fnc-freeze-mask',
        help='Post-hoc: (re)compute Data/frozen on an already-extracted fnc-* HDF5 file without '
             're-running the (expensive) pf2ens extraction - see fnc_plane.add_frozen_mask.',
    )
    fnc_freeze_mask.add_argument('h5_path', help='Path to an HDF5 file written by an fnc-* subcommand (>= 2 frames)')
    fnc_freeze_mask.add_argument('variable', help='Which Data/<variable> to judge frozen-ness from (e.g. \'vmag\')')
    fnc_freeze_mask.add_argument('--rel-threshold', type=float, default=0.01,
                                  help='Fraction of the observed dynamic range below which a point counts as '
                                       'frozen (default: 0.01)')
    fnc_freeze_mask.set_defaults(func=run_fnc_freeze_mask)

    fnc_plot = subparsers.add_parser(
        'fnc-plot',
        help='Quick-look contour plot from an already-extracted fnc-meridional HDF5 file '
             '(no re-extraction - see fnc_plane.plot_frame).',
    )
    fnc_plot.add_argument('h5_path', help='Path to an HDF5 file written by `convert.py fnc-meridional`')
    fnc_plot.add_argument('variable', help='Which Data/<variable> to plot')
    fnc_plot.add_argument('output', help='Path to the output PNG')
    fnc_plot.add_argument('--frame', type=int, default=0,
                           help='Which frame_index to plot (default: 0, i.e. the first frame stored)')
    fnc_plot.set_defaults(func=run_fnc_plot)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
