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


def raw_positions_to_ensight_frame(positions: np.ndarray, axis_origin_raw: np.ndarray) -> np.ndarray:

    '''
    Shift raw .snc surfel positions (already scaled to meters, e.g.
    SNCReader.surfel_centroids() * lattice_scales['LatticeLength']) onto
    pf2ens's own coordinate convention, so the two can be matched
    point-for-point.

    pf2ens centers every face it's asked to export on lrf_axis_origin
    (the rotation axis's own point, in the .snc's raw/absolute
    coordinates) - i.e. pf2ens_position = raw_position - axis_origin_raw.
    An EARLIER version of this function instead subtracted the SELECTED
    mesh's own bounding-box midpoint ("pf2ens centers each axis
    independently on the mesh's own bounding-box midpoint") - that was
    wrong, just not detectably so on the single case it was validated
    against: a rotationally-symmetric (multi-blade/whole-rotor) face
    selection happens to have its own bbox center coincide with
    axis_origin almost exactly, so the two conventions were numerically
    indistinguishable there. They diverge for an ASYMMETRIC selection
    (e.g. one single blade out of several) - confirmed directly on a
    real one-blade case: EnsightFrame.positions() + axis_origin_raw
    reproduces the raw .snc's own absolute coordinates (matching to 5
    significant figures in the plane perpendicular to the rotation
    axis), while the blade's own bbox center sits about 0.2 m away from
    axis_origin_raw there - small in absolute terms, but enough to blow
    a ~0.17 m-radius blade's computed radius out to 50% past its actual
    tip when used as the reference point instead (and, since computed
    radius then mixes in the point's own chordwise offset too - see
    convert_snc_to_h5()'s Note - distorts the whole blade into a
    diagonal sliver rather than just shifting it). This function itself
    is never called directly in this module (the same shift is
    duplicated inline in convert_snc_to_h5()) - kept here as the
    documented reference for what that inline code does and why.

    Also confirmed on that same case: pf2ens's per-frame Geometry export
    does NOT rotate with the blade (two frames ~180 degrees of real LRF
    rotation apart produced byte-identical positions) - it's already in
    the same frame-independent convention the raw .snc's own Geometry
    uses, same as SNCReader.to_h5() never needing to rotate ITS Geometry
    either (only Surface_X/Y/Z-Force does, being a genuinely different,
    per-frame-measured global-frame quantity - see SNCReader's own "NOTE
    on reference frames"). No rotation correction of any kind is needed
    or applied here.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Raw .snc positions, in the SAME absolute/raw coordinates
        SNCReader.surfel_centroids() (scaled to meters) returns - NOT
        yet shifted onto pf2ens's own convention.
    axis_origin_raw : np.ndarray, shape (3,)
        SNCReader.lrf_axis_origin, scaled to meters (lrf_axis_origin *
        lattice_scales['LatticeLength']) - the SAME point pf2ens itself
        subtracts internally.

    Returns
    -------
    np.ndarray, shape (N, 3)
    '''

    return positions - axis_origin_raw


class EnsightFrame:

    '''
    A single pf2ens single-frame EnSight Gold export, i.e. the output of:

    pf2ens -f <frame> -b <basename> <measurement_file>.snc

    All variables come out already in real MKS units (pf2ens's default),
    and positions/normals are exact for this specific frame (no rotation
    reconstruction involved - this describes DATA like pressure, which
    genuinely is specific to that frame's own instant). positions()
    itself does NOT rotate between frames - confirmed directly (two
    frames ~180 degrees of real LRF rotation apart produced byte-
    identical positions): it's in the same frame-independent convention
    the raw .snc's own Geometry uses, just shifted onto pf2ens's own
    lrf_axis_origin-relative coordinate convention (see
    raw_positions_to_ensight_frame()) - no rotation correction needed or
    applied anywhere in this module.
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

        pf2ens writes ONE BLOCK PER INCLUDED FACE (see convert_snc_to_h5's
        face_names), not one block total - confirmed on a real 3-face case
        (blade1/blade2/hub): `multiblock[0]` alone silently kept only the
        FIRST face and dropped the other two, with no error anywhere (this
        is what was actually still wrong after convert_snc_to_h5 was fixed
        to pass all present faces to pf2ens's `-i` - pf2ens was correctly
        including everything, this method was the one throwing data away
        afterward). combine() merges every block into one mesh; a case
        that only ever had one block (the original isolated-rotor case,
        one lumped face) is unaffected either way.
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
            self._mesh = multiblock.combine(merge_points=False)

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

    def surface_split(self, positions: np.ndarray, reference_positions: np.ndarray,
                       reference_upper: np.ndarray) -> np.ndarray:

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
        positions : np.ndarray, shape (N, 3)
            This frame's OWN positions to classify - passed in explicitly
            (rather than calling self.positions() here) so the caller can
            pass the axis_origin-shifted version (see
            EnsightSeriesWriter.add_frame()) instead of this frame's raw
            pf2ens-convention positions - both reference_positions and
            this array need to be in the SAME (absolute, unshifted)
            coordinates for the nearest-neighbor match to land on the
            right points.
        reference_positions : np.ndarray, shape (M, 3)
            Raw .snc surfel positions, absolute/unshifted (SNCReader.
            surfel_centroids() * lattice_scales['LatticeLength']) - the
            SAME convention `positions` above must already be in.
        reference_upper : np.ndarray of bool, shape (M,)
            SNCReader.surface_split() on that same file.
        '''

        tree = cKDTree(reference_positions)
        _, idx = tree.query(positions)
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
                 reference_positions=None, reference_upper=None, axis_origin=None, axis_direction=None,
                 blade_lrf_offset_deg: float = 0.0):

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
        self.blade_lrf_offset_deg = blade_lrf_offset_deg
        self._h5f = h5py.File(output_path, 'w')
        self._data_groups = None
        self._masks = None
        self._meta_group = None
        self._vector_variables = set()
        self._n_frames = 0

    def _init_datasets(self, frame: EnsightFrame, n_points: int, positions: np.ndarray):

        variables = self.variables or frame.variable_names()
        self.variables = variables
        self._vector_variables = set()

        if self.surface_split:
            upper = frame.surface_split(positions, self.reference_positions, self.reference_upper)
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
        self._meta_group.attrs['blade_lrf_offset_deg'] = self.blade_lrf_offset_deg

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

        frame.positions() is shifted by +self.axis_origin before use
        (undoing pf2ens's own internal lrf_axis_origin-centering - see
        raw_positions_to_ensight_frame()), whenever self.axis_origin is
        given - so Geometry/surface_split both end up in the SAME
        absolute coordinates self.axis_origin/self.axis_direction (and
        the raw .snc's own Geometry) are expressed in. No per-frame
        rotation correction - pf2ens's Geometry doesn't rotate between
        frames (confirmed empirically - see
        raw_positions_to_ensight_frame()'s docstring), unlike
        Surface_X/Y/Z-Force in the forces branch.

        self.blade_lrf_offset_deg, if nonzero, additionally rotates ONLY
        the stored Geometry (is_reference, below) about
        self.axis_direction through self.axis_origin, by the SAME
        constant angle/sign convention as SNCReader.to_h5()'s own
        blade_lrf_offset_deg - a FIXED mounting/modeling misalignment
        between the LRF's own nominal zero-orientation and the blade's
        actual geometric orientation, independent of which branch
        (forces or pressure) is doing the reading. Applied AFTER
        surface_split's own nearest-neighbor match (which uses the
        UNROTATED positions - rotating first would only misalign it
        against reference_positions, which come from the SAME, still-
        unrotated raw .snc convention), so classification is unaffected
        by this purely cosmetic/orientation correction.
        '''

        positions = frame.positions()
        if self.axis_origin is not None:
            positions = positions + self.axis_origin

        n_points = positions.shape[0]

        if self._data_groups is None:
            self._init_datasets(frame, n_points, positions)

        if is_reference:
            geo_positions = positions
            if self.blade_lrf_offset_deg != 0.0:
                axis = self.axis_direction / np.linalg.norm(self.axis_direction)
                blade_offset_rad = np.radians(self.blade_lrf_offset_deg)
                geo_positions = SNCReader._rotate_about_axis(
                    geo_positions - self.axis_origin, axis, -blade_offset_rad) + self.axis_origin
            for label, mask in self._masks.items():
                geo = self._h5f.create_group(f'Geometry/{label}' if label else 'Geometry')
                geo.create_dataset('X', data=geo_positions[mask, 0].astype('f4'))
                geo.create_dataset('Y', data=geo_positions[mask, 1].astype('f4'))
                geo.create_dataset('Z', data=geo_positions[mask, 2].astype('f4'))
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
                       surface_split: bool = False, face_names: list = None,
                       blade_lrf_offset_deg: float = 0.0):

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
        for per-frame LRF_position/timing metadata (mid_s/lrf_position_rad,
        written into Metadata - used by SurfaceVariable.timetrace()/
        periodogram() for a real sampling rate). NOT needed for Geometry
        correctness - pf2ens's Geometry doesn't rotate between frames and
        needs no rotation correction (see raw_positions_to_ensight_frame()).
        If omitted, those Metadata fields are left blank.
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
    face_names : list of str, optional
        Which faces to pass to pf2ens's `-i/--include_faces` (e.g.
        `['/rotor::blade1', '/rotor::blade2', '/rotor::hub']`). None
        (default): auto-detect and include EVERY face that actually has
        surfel data in this .snc (via the raw `face` tag array, not just
        the full face_names/part_names catalog, which lists faces that
        may not have any measurement in THIS particular file) - printed
        for visibility either way (see below). Explicitly listing every
        present face, rather than omitting `-i` and trusting pf2ens's own
        default, matters on a mesh with more than one same-kind face
        (e.g. `/rotor::blade1` AND `/rotor::blade2` as SEPARATE named
        faces, not lumped into one shared face like this project's
        original isolated-rotor case) - confirmed on a real case that
        pf2ens's default (no `-i` at all) silently returned only ONE of
        the two blade faces, with no error, while SNCReader.to_h5() (which
        reads the raw per-surfel `face` tag directly and applies no such
        default) correctly captured both. Pass an explicit subset here if
        you only want specific face(s) (e.g. just one blade) instead of
        everything merged.
    blade_lrf_offset_deg : float
        Extra CONSTANT rotation [deg], about lrf_axis_direction through
        lrf_axis_origin, applied to the stored Geometry only - SAME
        parameter, same sign convention, same physical meaning as
        SNCReader.to_h5()'s own blade_lrf_offset_deg (a fixed mounting/
        modeling misalignment between the LRF's own nominal zero-
        orientation and the blade's actual geometric orientation - a
        property of the BLADE, not of which branch/tool is reading it,
        so use the SAME value here as whatever was used for this case's
        forces-branch conversion). 0 (no rotation) by default. See
        EnsightSeriesWriter.add_frame()'s docstring for exactly what
        this does and why it's applied after, not before,
        surface_split's own matching.

    Note
    ----
    Opens snc_path a second time (via SNCReader) regardless of
    surface_split, to get lrf_axis_origin/lrf_axis_direction for the
    output's Metadata, and to shift pf2ens's own output back onto the
    raw .snc's absolute coordinate convention (both branches then agree
    - see raw_positions_to_ensight_frame()). The shift used (written to
    Metadata as axis_origin) combines TWO sources, one analytic and one
    empirical, not a single formula:

    - PERPENDICULAR to the rotation axis: lrf_axis_origin directly - a
      fixed point ON the true rotation axis, correct regardless of
      which faces are selected (confirmed reliable even for an
      asymmetric, e.g. single-blade, selection - an earlier version of
      this code instead used the selected faces' own bounding-box
      midpoint uniformly for all 3 components, which only coincides
      with lrf_axis_origin for a symmetric, e.g. whole-rotor, selection,
      and was measurably wrong - a ~22cm offset - for an asymmetric
      one).
    - ALONG the rotation axis: measured EMPIRICALLY, from a direct
      comparison between pf2ens's own reference-frame output and the
      raw .snc's own bounding-box center for the same selected faces
      (see the code just before the main frame loop) - NOT assumed from
      the raw .snc's own bbox alone (a later, still-incorrect version of
      this code tried exactly that: close, but pf2ens's own internal
      re-meshing, splitting complex surfels into quads/trias for EnSight
      compatibility, shifts ITS OWN bbox center by a few mm relative to
      the raw surfel centroids - confirmed on a real case where that
      residual was comparable to the blade's own thickness and still
      corrupted surface_split's Upper/Lower match, even though the same
      bbox-based approach worked on a different case where the residual
      happened to be small enough not to matter). This is why pf2ens
      now runs for reference_frame BEFORE the main per-frame loop below,
      not inside it.
    '''

    reference_frame = first_frame if reference_frame is None else reference_frame

    frame_meta_by_index = parse_nc_stats(nc_stats_path) if nc_stats_path else {}

    cleanup_work_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix='ensight_to_h5_')

    ref_reader = SNCReader(snc_path)
    axis_direction = ref_reader.lrf_axis_direction
    axis_origin_raw = ref_reader.lrf_axis_origin * ref_reader.lattice_scales['LatticeLength']

    # Faces actually carrying surfel data in THIS .snc (not just every
    # face declared in the case's full geometry catalog - face_names has
    # entries, e.g. wind-tunnel walls/inlets, that never show up in the
    # raw per-surfel `face` tag array at all for a measurement file
    # scoped to just the rotor) - see face_names parameter docstring for
    # why this can't just be left to pf2ens's own default. Resolved
    # BEFORE bbox_center below (not after, as an earlier version of this
    # function had it) - see that block's comment for why the order
    # matters now.
    face_tag = ref_reader._f.variables['face'][:]
    present_face_ids = np.unique(face_tag)
    present_face_names = [ref_reader.face_names[i] for i in present_face_ids]

    if face_names is None:
        include_faces = present_face_names
    else:
        unknown = set(face_names) - set(present_face_names)
        if unknown:
            raise ValueError(
                f"face_names {sorted(unknown)} not present in '{snc_path}' - faces actually "
                f"present: {present_face_names}."
            )
        include_faces = face_names

    print(
        f"pf2ens will include {len(include_faces)} face(s) (of {len(present_face_names)} present "
        f"in this file): {include_faces}. Pass face_names=[...] to convert_snc_to_h5() "
        "(or --face-names on the command line) to restrict to a subset instead."
    )

    # reference_positions (for surface_split's nearest-neighbor match) are
    # the raw .snc's own ABSOLUTE positions, unshifted - the SAME
    # convention EnsightSeriesWriter.add_frame() brings pf2ens's own
    # positions back into (frame.positions() + axis_origin - see that
    # method and raw_positions_to_ensight_frame()), so the two point
    # clouds being matched are expressed consistently.
    include_face_ids = [i for i, name in enumerate(ref_reader.face_names) if name in include_faces]
    face_surfel_mask = np.isin(face_tag, include_face_ids)

    raw_positions = ref_reader.surfel_centroids() * ref_reader.lattice_scales['LatticeLength']
    raw_positions = raw_positions[face_surfel_mask]

    # axis_origin's PERPENDICULAR-to-axis component is lrf_axis_origin
    # directly (confirmed reliable even for an asymmetric, e.g. single-
    # blade, selection - it's a fixed point ON the true rotation axis,
    # independent of which faces happen to be selected). The ALONG-axis
    # component, though, is measured EMPIRICALLY below, from pf2ens's own
    # actual reference-frame output - NOT assumed from the raw .snc's own
    # bounding-box midpoint (an earlier version of this code did that,
    # and was close but not exact: pf2ens's own internal re-meshing
    # (splitting complex surfels into quads/trias for EnSight
    # compatibility - see EnsightFrame.normals()'s docstring) shifts its
    # OWN bbox center by a few mm relative to the raw surfel centroids,
    # confirmed on a real case where that residual was comparable to the
    # blade's own thickness and corrupted surface_split's Upper/Lower
    # match, even though the SAME approach worked on a different case
    # where the residual happened to be small enough not to matter).
    axis_direction_unit = axis_direction / np.linalg.norm(axis_direction)
    origin_along = (axis_origin_raw @ axis_direction_unit) * axis_direction_unit
    origin_perp = axis_origin_raw - origin_along
    raw_bbox_center = (raw_positions.min(axis=0) + raw_positions.max(axis=0)) / 2

    if surface_split:
        reference_positions = raw_positions
        reference_upper = ref_reader.surface_split()[face_surfel_mask]
    else:
        reference_positions = reference_upper = None

    ref_reader.close()

    # Absolute, resolved BEFORE pf2ens runs with cwd=work_dir - snc_path
    # may have been given relative to the caller's own working directory,
    # which is no longer where the subprocess runs.
    snc_path_abs = os.path.abspath(snc_path)

    # Run pf2ens for reference_frame FIRST (before the main loop below),
    # so axis_origin's along-axis component can be measured from its
    # ACTUAL output (see above) before axis_origin is finalized -
    # EnsightSeriesWriter needs the complete axis_origin up front, since
    # it applies to every frame via add_frame(). The resulting
    # EnsightFrame is reused for reference_frame's own turn in the main
    # loop instead of running pf2ens for it a second time.
    ref_basename = f'frame_{reference_frame}'
    subprocess.run(
        ['pf2ens', '-f', str(reference_frame), '-b', ref_basename, '-i', ','.join(include_faces), snc_path_abs],
        check=True, cwd=work_dir,
    )
    reference_ensight_frame = EnsightFrame(os.path.join(work_dir, ref_basename + '.case'))
    ref_positions = reference_ensight_frame.positions()
    pf2ens_bbox_center = (ref_positions.min(axis=0) + ref_positions.max(axis=0)) / 2
    empirical_shift = raw_bbox_center - pf2ens_bbox_center
    shift_along = (empirical_shift @ axis_direction_unit) * axis_direction_unit
    axis_origin = origin_perp + shift_along

    writer = EnsightSeriesWriter(
        output_path, variables=variables, surface_split=surface_split,
        reference_positions=reference_positions, reference_upper=reference_upper,
        axis_origin=axis_origin, axis_direction=axis_direction,
        blade_lrf_offset_deg=blade_lrf_offset_deg,
    )

    try:
        for frame in range(first_frame, last_frame + 1):

            if frame == reference_frame:
                ensight_frame = reference_ensight_frame
            else:
                # RELATIVE basename, run with cwd=work_dir - NOT
                # os.path.join(work_dir, ...) (an absolute path). pf2ens
                # writes whatever basename it's given straight into the
                # .case file as the geometry/variable filenames; if
                # that's already absolute, VTK's EnSight reader (which
                # expects a .case file's referenced filenames to be
                # relative to the .case file's own directory) blindly
                # joins its own directory onto them ANYWAY, producing a
                # doubled path like "/tmp/xxx//tmp/xxx/frame_1.geo.ens" -
                # confirmed exactly this failure mode on the HPC
                # (IndexError: index (0) out of range for this dataset,
                # from an empty multiblock after that doubled path
                # failed to open) - not a bad frame index, every frame
                # hit it the same way.
                basename = f'frame_{frame}'
                subprocess.run(
                    ['pf2ens', '-f', str(frame), '-b', basename, '-i', ','.join(include_faces), snc_path_abs],
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
    parser.add_argument('--face-names', default=None,
                         help='Comma-separated face names to pass to pf2ens (e.g. '
                              '"/rotor::blade1,/rotor::blade2") - default: every face present in '
                              'this .snc, printed at run time (see convert_snc_to_h5\'s face_names '
                              'docstring for why this is explicit rather than left to pf2ens\'s '
                              'own default).')
    args = parser.parse_args()

    convert_snc_to_h5(
        args.snc_path, args.output_path, args.first, args.last,
        nc_stats_path=args.nc_stats, reference_frame=args.reference_frame,
        work_dir=args.work_dir, surface_split=args.surface_split,
        face_names=args.face_names.split(',') if args.face_names else None,
    )


if __name__ == '__main__':
    main()
