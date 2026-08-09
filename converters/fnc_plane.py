import os
import re
import shutil
import subprocess
import tempfile
import numpy as np
import h5py

from converters.ensight_to_h5 import parse_nc_stats


'''
Convert a PowerFLOW .fnc volumetric fluid measurement file into an HDF5
plane/probe dataset, WITHOUT going through PowerVIZ.

Background (see rotaris' notes on this): .fnc volumetric files are a
multi-resolution (variable-resolution octree) point cloud spanning the
whole measurement volume - typically hundreds of millions of points - not
a tidy grid, and direct NetCDF reading (the way SNCReader reads .snc)
breaks down here: scipy.io.netcdf_file cannot even open a large single-
frame .fnc (its record-array reader assumes a variable's total byte size
fits in a C int, which large fluid volumes exceed).

A first version of this module drove `exaritool point-props.ri` /
`fluid-profile.ri` (interpolate an arbitrary point cloud directly against
the raw .fnc file) - abandoned after both tools proved unreliable at the
scale needed (crashes/internal errors for any query above ~2 points,
and even the 2-point case that "worked" gave numbers later found to
disagree by ~5 orders of magnitude with the validated method below).

Instead, this module drives `pf2ens` - the same tool rotaris already uses
for .snc Static Pressure (see ensight_to_h5.py) - which also accepts
.fnc fluid files directly: `pf2ens -f <frame> -v <vars> -b <basename>
<file>.fnc` exports one frame's FULL volumetric field to EnSight Gold.
That's loaded with `pyvista`, then sampled at our own query points via
VTK's cell-based interpolation (`pv.PolyData(points).sample(mesh)`),
which respects the volume's actual octree topology and reports per-point
validity (`vtkValidPointMask`) - that IS the mask, rather than a density-
threshold heuristic reconstructed after the fact from a scattered point
cloud. Validated: sampled values agreed with exaritool's (non-`-interp`)
nearest-neighbor result at the same location; `nc-stats.ri`'s own MKS
bounding box matched the loaded mesh's bounds exactly.

`pf2ens` and `exaritool` (still used here for metadata - see
run_nc_stats) must be on $PATH - source the PowerFLOW environment first:

    source /project/rrg-moreaust-ac/Env/powerflow_env.sh 6-2025-R3

`pyvista`/`vtk` must be importable - see rotaris' notes on the working
module recipe for this (`module load StdEnv/2023 gcc/12.3 python/3.11
vtk/9.4.2 scipy-stack/2023b` - python/3.11 must load before vtk/9.4.2 -
then `source ~/rotaris-venv/bin/activate`; plain `pip install pyvista`
alone is not enough, `vtk` itself needs the module). IMPORTANT: never
pipe a `module load` line through anything (`| tail`, `| grep`, ...) -
that runs it in a subshell and its environment changes are silently lost
in the parent shell.

Three ways to define the query point cloud (the "geometry"):

- meridional_plane_points(): a hub-to-tip, inlet-to-outlet rectangle in
  the plane containing the rotor axis, rotated to a chosen azimuthal
  angle - this is the "radial-cuts" shape used for vortex tracking, but
  built directly (no PowerVIZ reference-geometry STL needed).
- iso_radius_points(): a fixed-radius cylindrical surface, unrolled into
  (azimuth, axial) - the "span-cuts" (e.g. 75% span) shape.
- Or just build/load your own (N, 3) point array (e.g. probe locations)
  and pass it straight to extract_to_h5().

Both generators default their extents to the file's own measurement
bounding box (from nc-stats.ri) when not given explicitly - a SAFE OUTER
BOUND, not the true duct/blade footprint. Grid points outside the real
fluid domain come back with valid=False (see FNCVolumeFrame.sample) -
that is the mask, reported by PowerFLOW/VTK itself, rather than a
density-threshold heuristic reconstructed after the fact from a
scattered point cloud.

Cost: pf2ens exports the FULL volume per frame regardless of how many
query points you actually want - ~2m46s and ~11GB geometry + ~1GB per
variable, for the 250M-point case this was validated against. Multiple
variables in one call amortize the (fixed, large) geometry cost; many
frames do not - extracting a real production grid across many frames
should run as a SLURM batch job (see run_conversion.sh's existing
pattern), not interactively.
'''


def run_nc_stats(fnc_path: str, out_path: str) -> str:

    '''
    Run `exaritool nc-stats.ri <fnc_path> -detail`, saving its output to
    out_path (frame list, measurement bounding box, available variables -
    see parse_nc_stats, parse_bounding_box_mks, parse_variable_names).
    '''

    if os.path.exists(out_path):
        os.remove(out_path)  # exaritool refuses to overwrite an existing file

    subprocess.run(
        ['exaritool', 'nc-stats.ri', fnc_path, '-detail', '-out', out_path],
        check=True,
    )
    return out_path


def parse_bounding_box_mks(nc_stats_path: str) -> dict:

    '''
    Parse the physical-units (MKS) "Measurement bounding box" out of
    saved `exaritool nc-stats.ri -detail` output.

    Returns
    -------
    dict with 'min' and 'max', each np.ndarray shape (3,) - (x, y, z),
    meters.
    '''

    with open(nc_stats_path) as f:
        text = f.read()

    match = re.search(
        r'MKS:\s*\n\s*Min:\s*([\-\d.eE]+)\s+([\-\d.eE]+)\s+([\-\d.eE]+)\s*\n'
        r'\s*Max:\s*([\-\d.eE]+)\s+([\-\d.eE]+)\s+([\-\d.eE]+)',
        text,
    )

    if match is None:
        raise ValueError(f"Could not find an 'MKS: Min/Max' bounding box in {nc_stats_path}")

    values = [float(v) for v in match.groups()]
    return {'min': np.array(values[:3]), 'max': np.array(values[3:])}


def parse_variable_names(nc_stats_path: str) -> dict:

    '''
    Parse the "Variables present in file" (native) and "Derivable
    variables" lists out of saved `exaritool nc-stats.ri -detail` output.
    Either list is usable as a -prop argument to point-props.ri /
    fluid-profile.ri.

    Returns
    -------
    dict with keys 'native' and 'derivable', each a list of str.
    '''

    with open(nc_stats_path) as f:
        text = f.read()

    names = {}
    for key, label in (('native', 'Variables present in file'), ('derivable', 'Derivable variables')):
        match = re.search(rf'{label}:\s*\d+\s*\n((?:\s+\w+\s*\n)+)', text)
        if match is None:
            raise ValueError(f"Could not find a '{label}' list in {nc_stats_path}")
        names[key] = match.group(1).split()

    return names


def _rotate_about_axis(points: np.ndarray, axis: str, angle_deg: float) -> np.ndarray:

    '''Rotate an (N, 3) point array by angle_deg (degrees) about a global axis.'''

    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    axis = axis.lower()

    if axis == 'z':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'x':
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    return points @ R.T


def meridional_plane_points(angle_deg: float, axis: str = 'z', n_inplane: int = 100, n_axial: int = 100,
                             inplane_range=None, axial_range=None, bbox: dict = None) -> tuple:

    '''
    Build a query grid for a meridional (hub-to-tip, inlet-to-outlet)
    plane cut at one azimuthal angle - the "radial-cuts" shape used for
    vortex tracking. A rectangle is built at angle=0 (spanning
    inplane_range along the in-plane axis, through the rotor axis, and
    axial_range along the rotor axis), then rotated about `axis` to
    angle_deg.

    Parameters
    ----------
    angle_deg : float
        Azimuthal angle, degrees.
    axis : str
        Rotor rotation axis ('x', 'y', or 'z').
    n_inplane, n_axial : int
        Grid resolution.
    inplane_range, axial_range : (float, float), optional
        Extent of the plane at angle=0. inplane_range is signed (e.g.
        (-R_tip, +R_tip), a full diameter through the hub, not a radius).
        Defaults to bbox's extent along the relevant axes if omitted -
        a safe outer bound (see module docstring), not the true duct
        footprint.
    bbox : dict, optional
        Output of parse_bounding_box_mks(); required if inplane_range or
        axial_range is omitted.

    Returns
    -------
    points : np.ndarray, shape (n_inplane * n_axial, 3)
    grid_shape : (int, int)
        (n_inplane, n_axial), for reshaping points/results back to a grid.
    '''

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    inplane_idx = [i for i in range(3) if i != axis_idx][0]

    if inplane_range is None or axial_range is None:
        if bbox is None:
            raise ValueError("bbox is required when inplane_range or axial_range is omitted")
        if inplane_range is None:
            inplane_range = (bbox['min'][inplane_idx], bbox['max'][inplane_idx])
        if axial_range is None:
            axial_range = (bbox['min'][axis_idx], bbox['max'][axis_idx])

    u = np.linspace(inplane_range[0], inplane_range[1], n_inplane)
    a = np.linspace(axial_range[0], axial_range[1], n_axial)
    U, A = np.meshgrid(u, a, indexing='ij')

    points = np.zeros((U.size, 3))
    points[:, inplane_idx] = U.ravel()
    points[:, axis_idx] = A.ravel()

    return _rotate_about_axis(points, axis, angle_deg), (n_inplane, n_axial)


def iso_radius_points(radius: float, axis: str = 'z', n_theta: int = 360, n_axial: int = 100,
                       theta_range=(0.0, 360.0), axial_range=None, bbox: dict = None) -> tuple:

    '''
    Build a query grid for a fixed-radius cylindrical surface, unrolled
    into (azimuth, axial) - the "span-cuts" (e.g. 75% span) shape.

    Parameters
    ----------
    radius : float
        Physical radius, meters.
    axis : str
        Rotor rotation axis ('x', 'y', or 'z').
    n_theta, n_axial : int
        Grid resolution.
    theta_range : (float, float)
        Azimuthal range, degrees.
    axial_range : (float, float), optional
        Defaults to bbox's extent along `axis` if omitted.
    bbox : dict, optional
        Output of parse_bounding_box_mks(); required if axial_range is
        omitted.

    Returns
    -------
    points : np.ndarray, shape (n_theta * n_axial, 3)
    grid_shape : (int, int)
        (n_theta, n_axial), for reshaping points/results back to a grid.
    '''

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    other_idx = [i for i in range(3) if i != axis_idx]

    if axial_range is None:
        if bbox is None:
            raise ValueError("bbox is required when axial_range is omitted")
        axial_range = (bbox['min'][axis_idx], bbox['max'][axis_idx])

    theta = np.deg2rad(np.linspace(theta_range[0], theta_range[1], n_theta))
    a = np.linspace(axial_range[0], axial_range[1], n_axial)
    Theta, A = np.meshgrid(theta, a, indexing='ij')

    points = np.zeros((Theta.size, 3))
    points[:, other_idx[0]] = radius * np.cos(Theta.ravel())
    points[:, other_idx[1]] = radius * np.sin(Theta.ravel())
    points[:, axis_idx] = A.ravel()

    return points, (n_theta, n_axial)


def run_pf2ens(fnc_path: str, frame: int, variables: list, basename: str):

    '''
    Run `pf2ens` for one frame of a .fnc fluid measurement file, exporting
    the FULL volumetric field (every measurement point, not a subset) to
    EnSight Gold - the same tool rotaris already uses for .snc Static
    Pressure (see ensight_to_h5.py), which also accepts .fnc directly.

    All variables requested are exported in one call so the (large -
    ~11GB for the 250M-point case this was validated against) geometry
    file is only written once per frame, not once per (frame, variable).

    Parameters
    ----------
    fnc_path : str
    frame : int
        Frame index to export (matches nc-stats.ri's frame numbering).
    variables : list of str
        pf2ens's own short variable codes (e.g. 'vmag', 'p', 'vx', 'rho',
        'temp', 'tke', 'eps', 'stm', 'e', 'ei', 'ek', 'en', 'pt', 'ptd')
        - run `pf2ens -d <fnc_path>` to list them for a given file. Note
        these are NOT the same strings as exaritool's snake_case names
        (parse_variable_names) - the EnSight arrays that come back ARE
        named descriptively though (e.g. 'velocity_magnitude' for 'vmag'),
        matching exaritool's convention.
    basename : str
        Passed to pf2ens -b; output files are '<basename>.case',
        '<basename>.geo.ens', and one '<basename>.<var>.ens' per variable.

    Returns
    -------
    str
        Path to the written '<basename>.case' file.
    '''

    cmd = ['pf2ens', '-f', str(frame), '-v', ','.join(variables), '-b', basename, fnc_path]
    subprocess.run(cmd, check=True)
    return basename + '.case'


class FNCVolumeFrame:

    '''
    A single pf2ens full-volume EnSight Gold export of one .fnc frame
    (see run_pf2ens) - the fluid-file analogue of EnsightFrame in
    ensight_to_h5.py, except this one is sampled onto arbitrary query
    points ourselves (via pyvista/VTK cell-based interpolation) rather
    than read point-for-point, since the underlying mesh (hundreds of
    millions of points, multi-resolution octree) is never meant to be
    consumed in full downstream.
    '''

    def __init__(self, case_path: str):
        self.case_path = case_path
        self._mesh = None

    def mesh(self):

        '''
        Load (and cache) the volumetric mesh via pyvista.

        pv.get_reader() mis-resolves a .case file whose internal geometry
        line is an absolute path (as pf2ens writes, given an absolute -b
        basename) against the reader's own base directory, doubling the
        path and failing to open it - chdir into the case file's own
        directory and pass a bare filename to sidestep this.
        '''

        if self._mesh is None:
            import pyvista as pv

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

    def sample(self, points: np.ndarray) -> dict:

        '''
        Interpolate every array in this frame's mesh at `points`, via
        VTK's cell-based interpolation (respects the volume's actual
        multi-resolution octree topology, unlike a nearest/linear fit
        reconstructed from a scattered point cloud after the fact).

        Returns
        -------
        dict[str, np.ndarray]
            One entry per mesh array (pf2ens's descriptive names, e.g.
            'velocity_magnitude'), plus 'valid': bool array, True where
            the query point actually fell inside the measurement volume
            (VTK's vtkValidPointMask) - this IS the mask; no density-
            threshold heuristic needed.
        '''

        import pyvista as pv

        mesh = self.mesh()
        sampled = pv.PolyData(points).sample(mesh)

        result = {name: np.asarray(sampled[name]) for name in mesh.array_names}
        result['valid'] = np.asarray(sampled['vtkValidPointMask']).astype(bool)
        return result


def _compute_frozen_mask(values: np.ndarray, valid: np.ndarray, rel_threshold: float) -> np.ndarray:

    '''
    Flag points whose value barely changes across frames, relative to how
    much real fluid in this same dataset typically changes - the
    signature of a lattice cell inside solid geometry that PowerFLOW's
    .fnc measurement volume didn't exclude (see fnc_plane's module notes:
    confirmed on a body-fixed stator hub that `vtkValidPointMask` failed
    to mask out - such cells sit frozen at their initial condition
    forever, since the solver never updates them, while every real fluid
    point - even a slow one - keeps moving at least a little frame to
    frame).

    Parameters
    ----------
    values : np.ndarray, shape (n_frames, n_points)
    valid : np.ndarray of bool, shape (n_frames, n_points)
        PowerFLOW/VTK's own mask (see FNCVolumeFrame.sample) - only
        frames where a point was already valid are used to judge whether
        it's frozen.
    rel_threshold : float
        A point is flagged frozen if its own range (max - min) across the
        frames it's valid in is below `rel_threshold * global_range`,
        where global_range is the max-min of every valid (frame, point)
        value in this dataset - i.e. a fraction of this extraction's own
        observed dynamic range, not an absolute physical unit. This is a
        heuristic, not exact - tune rel_threshold per case/variable if it
        over- or under-flags (compare against a region you know is real
        fluid vs. one you know is solid, the way this was validated).

    Returns
    -------
    np.ndarray of bool, shape (n_points,)
        True where the point looks frozen (has >= 2 valid frames and its
        range across them falls below the threshold). Points with fewer
        than 2 valid frames are left False (not enough information to
        judge - trust the original valid mask as-is).
    '''

    n_frames, n_points = values.shape
    valid_bool = valid.astype(bool)

    all_valid_values = values[valid_bool]
    if all_valid_values.size == 0:
        return np.zeros(n_points, dtype=bool)
    global_range = all_valid_values.max() - all_valid_values.min()
    threshold = rel_threshold * global_range

    frozen = np.zeros(n_points, dtype=bool)
    n_valid_frames = valid_bool.sum(axis=0)

    for i in np.where(n_valid_frames >= 2)[0]:
        v = values[valid_bool[:, i], i]
        frozen[i] = (v.max() - v.min()) < threshold

    return frozen


def add_frozen_mask(h5_path: str, variable: str, rel_threshold: float = 0.01):

    '''
    Post-hoc version of the frozen-point masking extract_to_h5() can do
    inline (see freeze_mask_variable) - reads Data/<variable> and
    Data/valid from an already-extracted HDF5 file (needs >= 2 frames),
    computes _compute_frozen_mask(), and writes the result to
    Data/frozen (overwriting it if already present). Useful to try a
    different variable/threshold without re-running the (expensive)
    pf2ens extraction.
    '''

    with h5py.File(h5_path, 'r+') as f:
        values = f[f'Data/{variable}'][:]
        valid = f['Data/valid'][:]

        if values.shape[0] < 2:
            raise ValueError(f"'{h5_path}' only has {values.shape[0]} frame(s) - need >= 2 to judge frozen points")

        frozen = _compute_frozen_mask(values, valid, rel_threshold)

        if 'frozen' in f['Data']:
            del f['Data/frozen']
        f['Data'].create_dataset('frozen', data=frozen)
        f['Data/frozen'].attrs['mask_variable'] = variable
        f['Data/frozen'].attrs['rel_threshold'] = rel_threshold

    print(f"wrote Data/frozen to '{h5_path}' ({frozen.sum()}/{len(frozen)} points flagged, "
          f"variable='{variable}', rel_threshold={rel_threshold})")


def extract_to_h5(fnc_path: str, points: np.ndarray, variables: list, output_path: str,
                   frames: list = None, grid_shape: tuple = None, geometry_attrs: dict = None,
                   freeze_mask_variable: str = None, freeze_rel_threshold: float = 0.01,
                   work_dir: str = None, nc_stats_path: str = None):

    '''
    For every requested frame, export the full volume via pf2ens
    (run_pf2ens), sample it at `points` (FNCVolumeFrame.sample - proper
    cell-based interpolation, not a heuristic mask), append the result
    into one growing HDF5 file, then delete that frame's intermediate
    EnSight export (large - dominated by the ~11GB geometry file in the
    case this was validated against) before moving to the next frame -
    same incremental-write-then-delete pattern convert_snc_to_h5() /
    EnsightSeriesWriter already use for .snc Static Pressure.

        Geometry/X, Y, Z         the query points, shape (n_points,)
        Data/valid               shape (n_frames, n_points), bool - True
                                  where the query point fell inside the
                                  measurement volume for that frame (from
                                  vtkValidPointMask). NOT a complete solid-
                                  geometry mask by itself - confirmed it
                                  can miss body-fixed solid parts (see
                                  freeze_mask_variable below).
        Data/<variable>          shape (n_frames, n_points), one dataset
                                  per requested variable (pf2ens's
                                  descriptive array names, e.g.
                                  'velocity_magnitude')
        Data/frozen               shape (n_points,), bool - only written if
                                  freeze_mask_variable is given and >= 2
                                  frames were extracted (see
                                  _compute_frozen_mask/add_frozen_mask) -
                                  True where the point's value barely
                                  changes across frames relative to real
                                  fluid elsewhere in this extraction, the
                                  signature of a lattice cell inside solid
                                  geometry that vtkValidPointMask didn't
                                  exclude. Combine with Data/valid
                                  (valid & ~frozen) for the best available
                                  mask.
        Metadata/frame_index, start_ts, end_ts, mid_ts, mid_s,
                 lrf_position_rad
                                  shape (n_frames,), from nc-stats.ri
                                  (parse_nc_stats), row i describes
                                  Data's row i

    Parameters
    ----------
    fnc_path : str
        Path to the PowerFLOW fluid measurement file (.fnc).
    points : np.ndarray, shape (n_points, 3)
        Query point cloud, e.g. from meridional_plane_points() or
        iso_radius_points(), or your own array.
    variables : list of str
        pf2ens's short variable codes (see run_pf2ens) - run
        `pf2ens -d <fnc_path>` to list them for this file.
    output_path : str
        Path to the HDF5 file to create.
    frames : list of int, optional
        Which frame indices to extract (default: every frame nc-stats.ri
        reports). Each frame costs one pf2ens call - a full-volume
        export, several minutes and several GB for a large case - so
        extracting many frames is expensive; for a real production grid,
        run this as a SLURM batch job (see run_conversion.sh's existing
        pattern), not interactively.
    grid_shape : tuple, optional
        If points came from a grid generator, its returned grid_shape -
        stored as a Geometry attribute so the flat point list can be
        reshaped back into a 2D grid later.
    geometry_attrs : dict, optional
        Extra scalar metadata to stash as Geometry attrs (e.g.
        {'kind': 'meridional', 'angle_deg': 30.0, 'axis': 'z'}) - lets
        plot_frame() (meridional plane only, so far) reconstruct the
        un-rotated (in-plane, axial) coordinates for a correct 2D plot.
    freeze_mask_variable : str, optional
        If given (and >= 2 frames are extracted), compute Data/frozen
        from this variable's values across frames (see
        _compute_frozen_mask) - flags points that look frozen at their
        initial condition (solid geometry vtkValidPointMask missed)
        rather than genuinely varying fluid. A variable with a lot of
        real unsteadiness where you expect fluid (e.g. a velocity
        component, not something inherently near-constant like density
        in low-Mach flow) works best - validated with 'vmag'.
    freeze_rel_threshold : float
        See _compute_frozen_mask - fraction of this extraction's own
        observed dynamic range below which a point counts as frozen
        (default 0.01, i.e. 1%). Only used if freeze_mask_variable is set.
    work_dir : str, optional
        Directory for intermediate pf2ens output (default: an
        auto-cleaned temp directory).
    nc_stats_path : str, optional
        Path to already-saved `exaritool nc-stats.ri -detail` output. If
        omitted, nc-stats.ri is run fresh.
    '''

    cleanup_work_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix='fnc_plane_')

    try:
        if nc_stats_path is None:
            nc_stats_path = run_nc_stats(fnc_path, os.path.join(work_dir, 'nc_stats.txt'))

        frame_meta = parse_nc_stats(nc_stats_path)

        points = np.asarray(points, dtype=float)
        n_points = len(points)

        frames = sorted(frame_meta.keys()) if frames is None else sorted(frames)

        h5f = h5py.File(output_path, 'w')

        try:
            geo = h5f.create_group('Geometry')
            geo.create_dataset('X', data=points[:, 0].astype('f4'))
            geo.create_dataset('Y', data=points[:, 1].astype('f4'))
            geo.create_dataset('Z', data=points[:, 2].astype('f4'))
            if grid_shape is not None:
                geo.attrs['grid_shape'] = grid_shape
            for key, value in (geometry_attrs or {}).items():
                geo.attrs[key] = value

            data_group = h5f.create_group('Data')
            valid_ds = data_group.create_dataset(
                'valid', shape=(0, n_points), maxshape=(None, n_points), dtype='bool'
            )
            var_ds = {
                var: data_group.create_dataset(var, shape=(0, n_points), maxshape=(None, n_points), dtype='f4')
                for var in variables
            }

            meta = h5f.create_group('Metadata')
            meta_ds = {}
            for key in ('frame_index', 'start_ts', 'end_ts'):
                meta_ds[key] = meta.create_dataset(key, shape=(0,), maxshape=(None,), dtype='i8')
            for key in ('mid_ts', 'mid_s', 'lrf_position_rad'):
                meta_ds[key] = meta.create_dataset(key, shape=(0,), maxshape=(None,), dtype='f8')

            for frame in frames:

                print(f'Converting frame {frame} via pf2ens ({len(variables)} variables)...')
                basename = os.path.join(work_dir, f'frame_{frame}')
                case_path = run_pf2ens(fnc_path, frame, variables, basename)

                sampled = FNCVolumeFrame(case_path).sample(points)
                missing = [var for var in variables if var not in sampled and _long_name(var) not in sampled]
                if missing:
                    raise ValueError(
                        f"pf2ens/pyvista returned arrays {list(sampled.keys())} for frame {frame}, "
                        f"missing requested variable(s) {missing}"
                    )

                row = valid_ds.shape[0]
                valid_ds.resize(row + 1, axis=0)
                valid_ds[row] = sampled.get('valid', np.ones(n_points, dtype=bool))

                for var in variables:
                    values = sampled.get(var, sampled.get(_long_name(var)))
                    var_ds[var].resize(row + 1, axis=0)
                    var_ds[var][row] = values

                fm = frame_meta.get(frame, {})
                for key in ('frame_index', 'start_ts', 'end_ts'):
                    meta_ds[key].resize(row + 1, axis=0)
                for key in ('mid_ts', 'mid_s', 'lrf_position_rad'):
                    meta_ds[key].resize(row + 1, axis=0)

                meta_ds['frame_index'][row] = frame
                meta_ds['start_ts'][row] = fm.get('start_ts', 0)
                meta_ds['end_ts'][row] = fm.get('end_ts', 0)
                meta_ds['mid_ts'][row] = fm.get('mid_ts', np.nan)
                meta_ds['mid_s'][row] = fm.get('mid_s', np.nan)
                meta_ds['lrf_position_rad'][row] = fm.get('lrf_position_rad', np.nan)

                for fname in os.listdir(work_dir):
                    if fname.startswith(f'frame_{frame}.'):
                        os.remove(os.path.join(work_dir, fname))

            if freeze_mask_variable is not None:
                if len(frames) < 2:
                    print(f"Only {len(frames)} frame(s) extracted - need >= 2 to compute Data/frozen, skipping.")
                else:
                    frozen = _compute_frozen_mask(var_ds[freeze_mask_variable][:], valid_ds[:], freeze_rel_threshold)
                    data_group.create_dataset('frozen', data=frozen)
                    data_group['frozen'].attrs['mask_variable'] = freeze_mask_variable
                    data_group['frozen'].attrs['rel_threshold'] = freeze_rel_threshold
                    print(f'Data/frozen: {frozen.sum()}/{n_points} points flagged as frozen '
                          f"(variable='{freeze_mask_variable}', rel_threshold={freeze_rel_threshold})")

        finally:
            h5f.close()

        print(f'wrote {output_path} ({n_points} points x {len(frames)} frames x {len(variables)} variables)')

    finally:
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


_PF2ENS_LONG_NAMES = {
    'vmag': 'velocity_magnitude', 'p': 'static_pressure', 'vx': 'x_velocity', 'vy': 'y_velocity',
    'vz': 'z_velocity', 'rho': 'density', 'temp': 'temperature', 'tke': 'turb_kinetic_energy',
    'eps': 'turb_dissipation', 'stm': 'strain_rate_magnitude', 'e': 'energy', 'ei': 'internal_energy',
    'ek': 'kinetic_energy', 'en': 'enthalpy', 'pt': 'total_pressure', 'ptd': 'total_pressure_deficit',
}


def _long_name(short_code: str) -> str:

    '''Best-effort mapping from a pf2ens short variable code to its EnSight array name.'''

    return _PF2ENS_LONG_NAMES.get(short_code, short_code)


def plot_frame(h5_path: str, variable: str, frame_index: int = 0, savepath: str = None,
               cmap: str = 'turbo', levels=100, ax=None):

    '''
    Quick-look filled-contour plot of one frame's field on its native
    2D grid - a sanity-check visualization, not the full post-processing
    pipeline (rotation-to-LRF/cylindrical-projection/phase-averaging is
    a separate stage, not implemented here).

    Only supports HDF5 files written by extract_to_h5() with a 2D grid
    (grid_shape set) from either meridional_plane_points()
    (geometry_attrs={'kind': 'meridional', 'angle_deg', 'axis'}) or
    iso_radius_points() (geometry_attrs={'kind': 'iso_radius', 'radius',
    'axis'}) - both written automatically by convert.py's
    fnc-meridional/fnc-iso-radius subcommands. For 'meridional', the
    stored angle_deg/axis are used to un-rotate the query points back to
    their local (in-plane, axial) coordinates before plotting. For
    'iso_radius', the points are re-expressed as (azimuth, axial) -
    both give axes that are meaningful regardless of which angle/radius
    was extracted.

    Points where Data/valid is False (outside the measurement volume for
    that frame - see FNCVolumeFrame.sample) are masked as NaN rather than
    plotted, instead of a density-threshold heuristic. If Data/frozen is
    present (see extract_to_h5's freeze_mask_variable / add_frozen_mask),
    those points are masked too - vtkValidPointMask alone can miss solid
    geometry that PowerFLOW's .fnc measurement volume didn't exclude
    (confirmed on a body-fixed stator hub).

    Parameters
    ----------
    h5_path : str
    variable : str
        Which Data/<variable> dataset to plot.
    frame_index : int
        Which Metadata/frame_index row to plot (default: 0, i.e. the
        first frame stored in the file - not necessarily the source
        file's frame 0, if a subset of frames was extracted).
    savepath : str, optional
        If given, the figure is saved here.
    cmap : str
    levels : int or array-like
        Passed to contourf.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure/axes is created if omitted.

    Returns
    -------
    (fig, ax)
    '''

    import matplotlib.pyplot as plt

    with h5py.File(h5_path, 'r') as f:

        geo = f['Geometry']
        if 'grid_shape' not in geo.attrs:
            raise ValueError(
                f"'{h5_path}' has no Geometry/grid_shape attribute - plot_frame() only works for "
                "output from a 2D grid generator (meridional_plane_points/iso_radius_points), not "
                "an arbitrary point cloud."
            )

        kind = geo.attrs.get('kind')
        grid_shape = tuple(int(n) for n in geo.attrs['grid_shape'])
        points = np.stack([geo['X'][:], geo['Y'][:], geo['Z'][:]], axis=1).astype(float)
        axis = str(geo.attrs['axis'])
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]

        if kind == 'meridional':
            angle_deg = float(geo.attrs['angle_deg'])
            local = _rotate_about_axis(points, axis, -angle_deg)
            inplane_idx = [i for i in range(3) if i != axis_idx][0]
            u = local[:, inplane_idx].reshape(grid_shape)
            a = local[:, axis_idx].reshape(grid_shape)
            xlabel, ylabel = 'in-plane [m]', 'axial [m]'
            title_extra = f'{angle_deg:g}deg'
        elif kind == 'iso_radius':
            other_idx = [i for i in range(3) if i != axis_idx]
            theta = np.degrees(np.arctan2(points[:, other_idx[1]], points[:, other_idx[0]]))
            u = theta.reshape(grid_shape)
            a = points[:, axis_idx].reshape(grid_shape)
            xlabel, ylabel = 'azimuth [deg]', 'axial [m]'
            title_extra = f"r={float(geo.attrs['radius']):g}m"
        else:
            raise ValueError(
                f"'{h5_path}' Geometry/kind={kind!r} - plot_frame() only supports 'meridional' or "
                "'iso_radius' (see extract_to_h5's geometry_attrs)."
            )

        frame_indices = f['Metadata/frame_index'][:]
        rows = np.where(frame_indices == frame_index)[0]
        if len(rows) == 0:
            raise ValueError(f"frame_index {frame_index} not found in '{h5_path}' (available: {frame_indices.tolist()})")
        row = rows[0]

        if variable not in f['Data']:
            raise ValueError(f"'{variable}' not found in '{h5_path}' Data (available: {list(f['Data'].keys())})")

        values = f[f'Data/{variable}'][row].astype(float).reshape(grid_shape)
        valid = f['Data/valid'][row].reshape(grid_shape)
        if 'frozen' in f['Data']:
            valid = valid & ~f['Data/frozen'][:].reshape(grid_shape)
        values[~valid] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
    else:
        fig = ax.figure

    contour = ax.contourf(u, a, values, levels=levels, cmap=cmap, extend='both')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(variable)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{variable}, frame {frame_index}, {title_extra}')
    if kind == 'meridional':
        ax.set_aspect('equal')  # both axes in meters - iso_radius mixes degrees and meters, leave auto
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200)

    return fig, ax
