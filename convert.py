import argparse

from converters.snc_reader import SNCReader
from converters.ensight_to_h5 import convert_snc_to_h5

'''
CLI entry point for the two .snc -> HDF5 conversion branches (see README.md):

    convert.py forces   ...   SNCReader.to_h5 - forces, skin friction, normals,
                               geometry, straight from the raw .snc. No pf2ens
                               needed.
    convert.py pressure ...   convert_snc_to_h5 - Static Pressure via pf2ens
                               (the only trusted source for pressure). Calls
                               pf2ens once per frame internally.

Meant to be run as a cluster batch job - see run_conversion.sh.
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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
