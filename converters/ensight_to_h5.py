import argparse
import os
import shutil
import subprocess
import tempfile
import numpy as np
import h5py
import pyvista as pv
from scipy.spatial import cKDTree

from converters.snc_reader import SNCReader, parse_nc_stats


def raw_positions_to_ensight_frame(positions: np.ndarray) -> np.ndarray:

    '''
    Re-center raw .snc surfel positions (already scaled to meters, e.g.
    SNCReader.surfel_centroids() * lattice_scales['LatticeLength']) onto
    pf2ens's own coordinate convention, so the two can be matched
    point-for-point.

    pf2ens centers each axis independently on the mesh's own bounding-box
    midpoint - NOT on lrf_axis_origin. The two only coincide for the two
    axes perpendicular to the rotation axis, and only because this rotor's
    geometry happens to be symmetric about that axis; the axis-aligned
    coordinate is offset by a further ~1cm relative to axis_origin on the
    case this was validated against, confirmed by comparing raw vs.
    pf2ens-reported bounding boxes directly.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)

    Returns
    -------
    np.ndarray, shape (N, 3)
    '''

    bbox_center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    return positions - bbox_center


class EnsightFrame:

    '''
    A single pf2ens single-frame EnSight Gold export, i.e. the output of:

    pf2ens -f <frame> -b <basename> <measurement_file>.snc

    All variables come out already in real MKS units (pf2ens's default),
    and positions/normals are exact for this specific frame (no rotation
    reconstruction involved).
    '''

    def __init__(self, case_path: str):

        self.case_path = case_path
        self._mesh = None

    def mesh(self):

        '''
        Load (and cache) the EnSight Gold mesh via pyvista.

        pv.get_reader() mis-resolves a .case file whose internal geometry
        line is an absolute path (as pf2ens writes, given an absolute -b
        basename) against the reader's own base directory, doubling the
        path and failing to open it - chdir into the case file's own
        directory and pass a bare filename to sidestep this (same fix as
        FNCVolumeFrame.mesh() in fnc_plane.py - confirmed on the HPC:
        IndexError: index (0) out of range for this dataset, from an
        empty multiblock after the doubled path failed to open).
        '''

        if self._mesh is None:
            case_dir = os.path.dirname(self.case_path) or '.'
            case_name = os.path.basename(self.case_path)
            cwd = os.getcwd()
            os.chdir(case_dir)
            try:
                reader = pv.get_reader(case_name)
                multiblock = reader.read()
            finally:
                os.chdir(cwd)
            self._mesh = multiblock[0]

        return self._mesh

    def variable_names(self):
        return list(self.mesh().cell_data.keys())

    def variable(self, name: str):

        '''Raw cell-data array for one variable (already real MKS units).'''

        return self.mesh().cell_data[name]

    def positions(self):

        '''Surfel positions (cell centers), shape (N, 3), meters.'''

        return self.mesh().cell_centers().points

    def normals(self):

        '''
        Surfel unit normals, shape (N, 3), computed geometrically from the
        mesh (pf2ens does not export normals directly).

        NOTE: validated to be unreliable for anything precision-sensitive,
        including sign-based classification (see surface_split()) - even
        with auto_orient_normals=True, cell-to-cell orientation within
        this mesh is only ~52% consistent with ground truth (near coin-
        flip), because pf2ens splits complex surfels into quads/trias for
        EnSight compatibility, which breaks the mesh connectivity that
        compute_normals() relies on to propagate a consistent orientation.
        Kept here only for storage/inspection purposes; do not use for
        classification - use surface_split() instead, which sidesteps
        this by borrowing the classification from the raw .snc file.
        '''

        surf = self.mesh().extract_surface()
        surf = surf.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
        return surf.cell_data['Normals']

    def surface_split(self, reference_positions: np.ndarray, reference_upper: np.ndarray) -> np.ndarray:

        '''
        Boolean mask over all cells, True for the "upper" surface,
        obtained by nearest-neighbor lookup against a trusted per-point
        classification computed on the raw .snc file
        (SNCReader.surface_split()).

        This exists because computing the split directly from this
        mesh's own (re-triangulated) normals is NOT reliable - see the
        note on normals(). Nearest-neighbor lookup against the raw
        file's own validated classification sidesteps that entirely.

        Parameters
        ----------
        reference_positions : np.ndarray, shape (M, 3)
            Raw .snc surfel positions, in pf2ens's coordinate convention
            - i.e. raw_positions_to_ensight_frame(SNCReader.surfel_centroids()
            * lattice_scales['LatticeLength']).
        reference_upper : np.ndarray of bool, shape (M,)
            SNCReader.surface_split() on that same file.
        '''

        tree = cKDTree(reference_positions)
        _, idx = tree.query(self.positions())
        return reference_upper[idx]


class EnsightSeriesWriter:

    '''
    Incrementally build one HDF5 file out of multiple EnsightFrame reads,
    one call to add_frame() per simulation frame. Geometry (positions
    only - see Normals note below) is stored once, taken from whichever
    frame is passed with is_reference=True - the blade's shape doesn't
    change between frames, only its orientation, and each frame's own
    geometry is only used to populate that one-time reference (see Note
    below).

    Normals
    -------
    Not written here. This writer exists for pf2ens-derived variables
    (chiefly Static Pressure), which don't need normals, and
    EnsightFrame.normals() is not reliable enough to propagate downstream
    for anything that does - see its docstring. Upper/lower classification
    should come from SNCReader.surface_split() instead (see surface_split
    below), not from any normal computed on this class's mesh.

    Note
    ----
    Because each EnsightFrame is independently exact for its own frame,
    you could also choose to store position per frame instead of once -
    more storage, but removes any reliance on the blade being a
    perfectly rigid, shape-invariant body. That's not implemented here;
    if disk space allows, storing coords per frame is the more foolproof
    option and would just mean writing to Geometry/X etc. inside
    add_frame() the same way Data variables are written.

    lrf_axis_origin / lrf_axis_direction (optional constructor args) are
    written to Metadata if given - matching SNCReader.to_h5()'s schema, so
    anything that needs the rotor's physical radius (e.g.
    bladeprocessor.SurfaceVariable._radius()) works the same way against
    either branch's output. convert_snc_to_h5() always supplies these
    (it already opens a SNCReader on the same .snc for the surface_split
    reference); pass them yourself if you construct this class directly
    without going through convert_snc_to_h5().
    '''

    def __init__(self, output_path: str, variables: list = None, surface_split: bool = False,
                 reference_positions=None, reference_upper=None, axis_origin=None, axis_direction=None):

        if surface_split and (reference_positions is None or reference_upper is None):
            raise ValueError(
                "reference_positions and reference_upper are required when "
                "surface_split=True - e.g. from SNCReader(same_snc_path) on the raw "
                "file (see raw_positions_to_ensight_frame() and SNCReader.surface_split())."
            )

        self.output_path = output_path
        self.variables = variables
        self.surface_split = surface_split
        self.reference_positions = reference_positions
        self.reference_upper = reference_upper
        self.axis_origin = axis_origin
        self.axis_direction = axis_direction
        self._h5f = h5py.File(output_path, 'w')
        self._data_groups = None
        self._masks = None
        self._meta_group = None
        self._vector_variables = set()
        self._n_frames = 0

    def _init_datasets(self, frame: EnsightFrame, n_points: int):

        variables = self.variables or frame.variable_names()
        self.variables = variables
        self._vector_variables = set()

        if self.surface_split:
            upper = frame.surface_split(self.reference_positions, self.reference_upper)
            self._masks = {'Upper': upper, 'Lower': ~upper}
        else:
            self._masks = {None: np.ones(n_points, dtype=bool)}

        self._data_groups = {}
        for label, mask in self._masks.items():
            n_label = int(mask.sum())
            group = self._h5f.create_group(f'Data/{label}' if label else 'Data')
            self._data_groups[label] = group

            for name in variables:
                key = name.replace(' ', '_')
                sample = frame.variable(name)

                if sample.ndim == 1:
                    group.create_dataset(
                        key, shape=(0, n_label), maxshape=(None, n_label), dtype='f4'
                    )
                elif sample.ndim == 2 and sample.shape[1] == 3:
                    self._vector_variables.add(name)
                    for suffix in ('X', 'Y', 'Z'):
                        group.create_dataset(
                            f'{key}_{suffix}', shape=(0, n_label), maxshape=(None, n_label),
                            dtype='f4'
                        )
                else:
                    raise ValueError(
                        f"Variable '{name}' has unsupported shape {sample.shape}; expected "
                        "(n_points,) for a scalar or (n_points, 3) for a vector."
                    )

        self._meta_group = self._h5f.create_group('Metadata')
        for key in ('frame_index', 'start_ts', 'end_ts'):
            self._meta_group.create_dataset(key, shape=(0,), maxshape=(None,), dtype='i8')
        for key in ('mid_ts', 'mid_s', 'lrf_position_rad'):
            self._meta_group.create_dataset(key, shape=(0,), maxshape=(None,), dtype='f8')
        if self.axis_origin is not None:
            self._meta_group.create_dataset('lrf_axis_origin', data=self.axis_origin)
        if self.axis_direction is not None:
            self._meta_group.create_dataset('lrf_axis_direction', data=self.axis_direction)

    def _append_row(self, dataset, value):
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    def add_frame(self, frame: EnsightFrame, frame_index: int, frame_meta: dict = None,
                  is_reference: bool = False):

        '''
        Append one frame's variables (and, if is_reference, the geometry)
        to the growing HDF5 file.

        Parameters
        ----------
        frame : EnsightFrame
        frame_index : int
            The simulation frame index this corresponds to (for metadata
            and for looking up frame_meta, if not passed explicitly).
        frame_meta : dict, optional
            One entry from parse_nc_stats()'s return value (start_ts,
            mid_ts, end_ts, mid_s, lrf_position_rad). If omitted, those
            fields are left as NaN/0.
        is_reference : bool
            If True, this frame's positions are stored as the file's
            Geometry (only meaningful the first time it's called).
            Normals are deliberately not written here - see
            EnsightFrame.normals().
        '''

        positions = frame.positions()
        n_points = positions.shape[0]

        if self._data_groups is None:
            self._init_datasets(frame, n_points)

        if is_reference:
            for label, mask in self._masks.items():
                geo = self._h5f.create_group(f'Geometry/{label}' if label else 'Geometry')
                geo.create_dataset('X', data=positions[mask, 0].astype('f4'))
                geo.create_dataset('Y', data=positions[mask, 1].astype('f4'))
                geo.create_dataset('Z', data=positions[mask, 2].astype('f4'))
                geo.attrs['reference_frame_index'] = frame_index

        for name in self.variables:
            key = name.replace(' ', '_')
            arr = frame.variable(name)

            for label, mask in self._masks.items():
                group = self._data_groups[label]
                if name in self._vector_variables:
                    for i, suffix in enumerate(('X', 'Y', 'Z')):
                        self._append_row(group[f'{key}_{suffix}'], arr[mask, i])
                else:
                    self._append_row(group[key], arr[mask])

        frame_meta = frame_meta or {}
        self._append_row(self._meta_group['frame_index'], frame_index)
        self._append_row(self._meta_group['start_ts'], frame_meta.get('start_ts', 0))
        self._append_row(self._meta_group['end_ts'], frame_meta.get('end_ts', 0))
        self._append_row(self._meta_group['mid_ts'], frame_meta.get('mid_ts', np.nan))
        self._append_row(self._meta_group['mid_s'], frame_meta.get('mid_s', np.nan))
        self._append_row(
            self._meta_group['lrf_position_rad'], frame_meta.get('lrf_position_rad', np.nan)
        )

        self._n_frames += 1

    def close(self):
        self._h5f.close()


def convert_snc_to_h5(snc_path: str, output_path: str, first_frame: int, last_frame: int,
                       nc_stats_path: str = None, variables: list = None,
                       reference_frame: int = None, work_dir: str = None,
                       surface_split: bool = False):

    '''
    Run the full pipeline for a range of frames within a single process:
    for each frame, call `pf2ens -f <frame>` to get a fresh, exact
    single-frame EnSight export, read it, append it into one growing
    HDF5 file, then delete the intermediate EnSight files before moving
    on to the next frame (each frame's raw geometry export is ~800 MB,
    so keeping more than one on disk at a time isn't necessary).

    Parameters
    ----------
    snc_path : str
        Path to the PowerFLOW surface measurement file (.snc).
    output_path : str
        Path to the combined HDF5 file to create.
    first_frame, last_frame : int
        Inclusive frame range to convert.
    nc_stats_path : str, optional
        Path to saved `exaritool nc-stats.ri <snc_path> -detail` output,
        for per-frame LRF_position/timing metadata. If omitted, that
        metadata is left blank.
    variables : list of str, optional
        Which cell-data variables to keep (default: all present).
    reference_frame : int, optional
        Which frame's geometry to store (default: first_frame).
    work_dir : str, optional
        Directory for intermediate pf2ens output (default: a temp dir,
        cleaned up automatically).
    surface_split : bool, optional
        If True, split into Upper/Lower surface groups using surfel
        classification borrowed from the raw .snc file, matched by
        nearest position (see EnsightFrame.surface_split and
        raw_positions_to_ensight_frame), by default False.

    Note
    ----
    Opens snc_path a second time (via SNCReader) regardless of
    surface_split, to copy lrf_axis_origin/lrf_axis_direction into the
    output's Metadata (see EnsightSeriesWriter's docstring), re-centered
    onto pf2ens's own bounding-box-midpoint convention the same way
    Geometry/X,Y,Z already is (see raw_positions_to_ensight_frame() -
    lrf_axis_origin is NOT pf2ens's coordinate origin, so this re-centering
    is required, not optional, for radius-from-axis_origin to come out
    correct downstream). This does read the full raw surfel cloud once
    (surfel_centroids()) to compute that shift, even when
    surface_split=False - not free, but avoids silently writing an
    axis_origin inconsistent with this file's own geometry.
    '''

    reference_frame = first_frame if reference_frame is None else reference_frame

    frame_meta_by_index = parse_nc_stats(nc_stats_path) if nc_stats_path else {}

    cleanup_work_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix='ensight_to_h5_')

    ref_reader = SNCReader(snc_path)
    axis_direction = ref_reader.lrf_axis_direction
    axis_origin_raw = ref_reader.lrf_axis_origin * ref_reader.lattice_scales['LatticeLength']

    # raw_positions_to_ensight_frame() re-centers positions on the mesh's own
    # bounding-box midpoint (pf2ens's convention, NOT lrf_axis_origin - see
    # that function's docstring). axis_origin needs the exact same shift to
    # stay consistent with the (already re-centered) Geometry/X,Y,Z this
    # writes - computed from the same raw surfel cloud either way, so this
    # read isn't wasted even when surface_split=False also needs it.
    raw_positions = ref_reader.surfel_centroids() * ref_reader.lattice_scales['LatticeLength']
    bbox_center = (raw_positions.min(axis=0) + raw_positions.max(axis=0)) / 2
    axis_origin = axis_origin_raw - bbox_center

    if surface_split:
        reference_positions = raw_positions - bbox_center
        reference_upper = ref_reader.surface_split()
    else:
        reference_positions = reference_upper = None

    ref_reader.close()

    writer = EnsightSeriesWriter(
        output_path, variables=variables, surface_split=surface_split,
        reference_positions=reference_positions, reference_upper=reference_upper,
        axis_origin=axis_origin, axis_direction=axis_direction,
    )

    # Absolute, resolved BEFORE the loop below starts running pf2ens with
    # cwd=work_dir - snc_path may have been given relative to the caller's
    # own working directory, which is no longer where the subprocess runs.
    snc_path_abs = os.path.abspath(snc_path)

    try:
        for frame in range(first_frame, last_frame + 1):

            # RELATIVE basename, run with cwd=work_dir - NOT
            # os.path.join(work_dir, ...) (an absolute path). pf2ens
            # writes whatever basename it's given straight into the
            # .case file as the geometry/variable filenames; if that's
            # already absolute, VTK's EnSight reader (which expects a
            # .case file's referenced filenames to be relative to the
            # .case file's own directory) blindly joins its own
            # directory onto them ANYWAY, producing a doubled path like
            # "/tmp/xxx//tmp/xxx/frame_1.geo.ens" - confirmed exactly
            # this failure mode on the HPC (IndexError: index (0) out of
            # range for this dataset, from an empty multiblock after
            # that doubled path failed to open) - not a bad frame index,
            # every frame hit it the same way.
            basename = f'frame_{frame}'

            subprocess.run(
                ['pf2ens', '-f', str(frame), '-b', basename, snc_path_abs],
                check=True, cwd=work_dir,
            )

            ensight_frame = EnsightFrame(os.path.join(work_dir, basename + '.case'))
            writer.add_frame(
                ensight_frame,
                frame_index=frame,
                frame_meta=frame_meta_by_index.get(frame),
                is_reference=(frame == reference_frame),
            )

            for fname in os.listdir(work_dir):
                if fname.startswith(f'frame_{frame}.') or fname.startswith(f'frame_{frame}\t'):
                    os.remove(os.path.join(work_dir, fname))

    finally:
        writer.close()
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def main():

    parser = argparse.ArgumentParser(
        description='Convert a range of PowerFLOW .snc frames (via pf2ens) into one HDF5 file.'
    )
    parser.add_argument('snc_path', help='Path to the PowerFLOW surface measurement file (.snc)')
    parser.add_argument('output_path', help='Path to the combined HDF5 output file')
    parser.add_argument('--first', type=int, required=True, help='First frame to convert')
    parser.add_argument('--last', type=int, required=True, help='Last frame to convert (inclusive)')
    parser.add_argument('--nc-stats', help='Path to saved `exaritool nc-stats.ri -detail` output')
    parser.add_argument('--reference-frame', type=int,
                         help='Frame whose geometry is stored (default: --first)')
    parser.add_argument('--work-dir', help='Directory for intermediate pf2ens output')
    parser.add_argument('--surface-split', action='store_true',
                         help='Split into Upper/Lower surface groups, classification borrowed '
                              'from the raw .snc file via nearest-neighbor matching (see '
                              'EnsightFrame.surface_split).')
    args = parser.parse_args()

    convert_snc_to_h5(
        args.snc_path, args.output_path, args.first, args.last,
        nc_stats_path=args.nc_stats, reference_frame=args.reference_frame,
        work_dir=args.work_dir, surface_split=args.surface_split,
    )


if __name__ == '__main__':
    main()
