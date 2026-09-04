import re
import warnings
from functools import reduce
from operator import mul
import h5py
import numpy as np
import scipy.io as sio
import scipy.io._netcdf as _nc


def parse_nc_stats(path: str):

    '''
    Parse the "Frame Start(ts) Mid(ts) End(ts) Mid(s) LRF_position(rad)"
    table out of the text output of:

    exaritool nc-stats.ri <file>.snc -detail > nc_stats.txt

    Lives here (not converters/ensight_to_h5.py, which also uses it for
    the pressure/pf2ens branch) since SNCReader.to_h5() needs it too, for
    an authoritative, PowerFLOW-computed alternative to _rotation_angle()'s
    self-derived formula - `ensight_to_h5.py` already imports SNCReader
    from this module, so defining it there instead would create a
    circular import.

    Parameters
    ----------
    path : str
        Path to the saved nc-stats.ri output.

    Returns
    -------
    dict[int, dict]
        Keyed by frame index, each value has keys:
        start_ts, mid_ts, end_ts, mid_s, lrf_position_rad.
    '''

    with open(path) as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Frame') and 'LRF_position' in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find a 'Frame ... LRF_position(rad)' table in {path}")

    frames = {}
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) < 6 or not re.match(r'^-?\d+$', parts[0]):
            break

        frame, start_ts, mid_ts, end_ts, mid_s, lrf_pos = parts[:6]
        frames[int(frame)] = {
            'start_ts': int(start_ts),
            'mid_ts': float(mid_ts),
            'end_ts': int(end_ts),
            'mid_s': float(mid_s),
            'lrf_position_rad': float(lrf_pos),
        }

    return frames


class _LargeRecordNetcdfFile(sio.netcdf_file):

    '''
    scipy.io.netcdf_file, patched for a record variable (one with a
    frame/record axis - this file's "measurements") whose TRUE per-record
    byte size exceeds what the classic NetCDF format's 32-bit `vsize`
    header field can hold - exactly what happens for a fine enough DNS
    mesh's Surface_*/Skin_Friction data (confirmed: a real case with an
    8 GB .snc file hit this, while a 5.4 GB one from the same project did
    not - it's the "measurements" record block's OWN byte size that
    matters, not overall file size, which is dominated by the separately-
    read, unaffected fixed-size geometry arrays).

    scipy's own source (scipy.io._netcdf.netcdf_file._read_var_array,
    verified directly against the installed scipy 1.10.1) documents the
    failure mode in a comment without actually handling it:

        "The 32-bit vsize field is not large enough to contain the size
        of variables that require more than 2^32 - 4 bytes, so 2^32 - 1
        is used in the vsize field for such variables."

    `vsize` is parsed with `_unpack_int`, a SIGNED 32-bit read
    (`frombuffer(..., '>i')`) - so that escape sentinel (0xFFFFFFFF) comes
    back as Python int -1, and scipy adds it straight into
    `self._recsize` (`self.__dict__['_recsize'] += vsize`) with no special
    -casing, corrupting it. The same signed/unsigned mismatch already
    misreads any LEGITIMATE (non-sentinel) vsize between 2^31 and
    2^32-2 bytes (~2.1-4.3 GB) as negative too, even without hitting the
    "officially" documented escape case. Either way, `self._recs *
    self._recsize` ends up negative, and `self.fp.read(negative_number)`
    raises `ValueError: read length must be non-negative or -1` - the
    exact error this fixes.

    Fix, part 1 (`self._recsize`): never trust the file's own `vsize`
    field when accumulating it - recompute the true value ourselves from
    the variable's own (correctly parsed, NOT subject to this 32-bit
    limit) shape/dtype instead, using the exact formula scipy's own
    comment documents (product of the non-record dimensions x itemsize,
    rounded up to the next multiple of 4).

    Fix, part 2 (NumPy's OWN, separate 32-bit ceiling): even with
    `_recsize` correct, scipy still reads every record variable (any
    NetCDF-level array whose first axis is the file's unlimited/frame
    dimension - "measurements" is one; this format apparently also has
    at least one other, smaller one riding the same frame axis, e.g. a
    per-frame timestamp/iteration value that isn't one of the physical
    quantities packed inside "measurements") through one NumPy
    STRUCTURED dtype: `dtypes = {'names': [...], 'formats': [...]}`, one
    "field" per record variable, each field a fixed-shape sub-array
    format string like '(500000, 3)>f4'. NumPy's structured-dtype
    machinery computes each field's byte size as a C `int` internally
    and raises `ValueError: invalid shape in fixed-type tuple: dtype
    size in bytes must fit into a C int` once a single field exceeds
    ~2^31 bytes (~2.1 GB) - confirmed on a real, single-frame, 8 GB .snc
    file: `_recsize` itself came out correct (part 1 worked), but
    building the structured dtype covering "measurements" (huge) plus
    whatever the other record variable is (small) still failed, because
    NumPy applies that ~2.1 GB ceiling per field, and one field
    ("measurements") is multiple GB regardless of how many OTHER,
    smaller fields sit alongside it.

    A plain (non-structured) ndarray has no such limit - its shape is
    stored as 64-bit npy_intp values, not a C int - so this override
    never builds a structured dtype for record variables at all,
    regardless of how many there are. Per the NetCDF classic-format
    spec, a file's record variables are interleaved by record (for each
    record r: each record variable's data, back-to-back, in declaration
    order, each padded to a 4-byte boundary) - and each variable's own
    `begin_` (its file offset, already correctly parsed - a genuine
    64-bit value, since a file large enough to need this fix is
    necessarily in the 64-bit-offset format variant) already tells us
    exactly where that variable's record-0 data starts. So: read the
    whole interleaved record block once as raw bytes, then hand each
    record variable its own plain strided view into it (stride between
    records = the corrected `self._recsize`; strides within one
    record = an ordinary C-contiguous layout for that variable's own
    shape). No NumPy dtype ever has to describe more than one variable's
    data at once, so the per-field ceiling above can't apply, no matter
    how many record variables the file has (an earlier version of this
    fix only bypassed the structured dtype when there was exactly one
    record variable - which turned out not to match this file's actual
    layout, and hit the exact same ceiling through the path left over
    for "more than one record variable").

    Everything else below is otherwise IDENTICAL to
    scipy.io._netcdf.netcdf_file._read_var_array - there is no smaller
    public hook to override, the fix has to happen inside this exact
    loop, so this necessarily duplicates scipy's private method rather
    than patching one line of it. Tied to scipy's private API by nature
    - if a scipy upgrade changes this method's internals, this override
    needs revisiting (it will fail loudly with an AttributeError
    pointing at whatever's missing, not silently misread data, since it
    calls straight through to the same private helpers scipy's own
    unmodified method does).
    '''

    def _read_var_array(self):

        header = self.fp.read(4)
        if header not in (_nc.ZERO, _nc.NC_VARIABLE):
            raise ValueError("Unexpected header.")

        begin = 0
        rec_vars = []
        rec_var_shapes = {}
        rec_var_dtypes = {}
        rec_var_begins = {}
        count = self._unpack_int()

        for i in range(count):

            (name, dimensions, shape, attributes,
             typecode, size, dtype_, begin_, vsize) = self._read_var()

            if shape and shape[0] is None:  # record variable

                rec_vars.append(name)
                rec_var_shapes[name] = shape
                rec_var_dtypes[name] = dtype_
                rec_var_begins[name] = begin_

                # NOT `vsize` (the file's own, possibly-corrupted field -
                # see class docstring) - recomputed independently from
                # shape/dtype instead. This formula already accounts for
                # the same 4-byte-boundary padding scipy's original code
                # only added explicitly for 'bch' typecodes (a no-op
                # round-up for everything else, since those itemsizes
                # are already multiples of 4).
                true_vsize = reduce(mul, shape[1:], 1) * size
                true_vsize = ((true_vsize + 3) // 4) * 4
                self.__dict__['_recsize'] += true_vsize

                if begin == 0:
                    begin = begin_

                data = None
            else:
                a_size = reduce(mul, shape, 1) * size
                if self.use_mmap:
                    data = self._mm_buf[begin_:begin_ + a_size].view(dtype=dtype_)
                    data.shape = shape
                else:
                    pos = self.fp.tell()
                    self.fp.seek(begin_)
                    data = _nc.frombuffer(self.fp.read(a_size), dtype=dtype_).copy()
                    data.shape = shape
                    self.fp.seek(pos)

            self.variables[name] = _nc.netcdf_variable(
                data, typecode, size, shape, dimensions, attributes,
                maskandscale=self.maskandscale)

        if rec_vars:

            if self.use_mmap:
                buffer = self._mm_buf
                block_start = 0  # self._mm_buf already spans the whole file
            else:
                pos = self.fp.tell()
                self.fp.seek(begin)
                buffer = self.fp.read(self._recs * self._recsize)
                block_start = begin
                self.fp.seek(pos)

            for name in rec_vars:

                shape = rec_var_shapes[name]
                shape_tail = shape[1:]
                item_dtype = np.dtype(rec_var_dtypes[name])

                tail_strides = []
                stride = item_dtype.itemsize
                for dim in reversed(shape_tail):
                    tail_strides.append(stride)
                    stride *= dim
                tail_strides.reverse()

                full_shape = (self._recs,) + shape_tail
                full_strides = (self._recsize,) + tuple(tail_strides)
                offset = rec_var_begins[name] - block_start

                arr = np.ndarray(
                    full_shape, dtype=item_dtype, buffer=buffer,
                    offset=offset, strides=full_strides,
                )
                self.variables[name].__dict__['data'] = arr


class SNCReader:

    '''
    Read a PowerFLOW .snc surface measurement file (NetCDF format) and
    expose its geometry, per-surfel face/part tags, and measured variables
    as plain NumPy arrays.

    This is a raw reader/converter only - it does not project the mesh
    onto a (radius, chord) grid, and it does not compute any derived
    post-processing quantities (e.g. wall shear/friction lines) - see
    bladeprocessor/ for tools that consume the HDF5 files this produces.
    It gives you surfel positions (as the centroid of each surfel's
    vertices), normals, areas, all measured variables, and the rotor's
    rotation axis/angular velocity, all in one place.

    NOTE on units: methods on this class (surfel_centroids(), variable(),
    lrf_axis_origin, etc.) return values exactly as stored in the file
    (PowerFLOW "lattice units"). to_h5() converts what it can to physical
    units before writing: positions/areas/lrf_axis_origin by
    lattice_scales['LatticeLength'] (m / m^2), and ForcePerArea-class
    variables (Surface X/Y/Z-Force, Skin Friction) by
    force_per_area_scale() (Pa) - the latter validated against this file's
    own stored Skin Friction (correlation ~0.999). Static Pressure is NOT
    converted - it needs PowerFLOW's internal Cp-based translation (see
    the PowerVIZ User's Guide, Appendix B), which isn't implemented here;
    use pf2ens for physical-unit pressure instead, and treat
    Data/Static_Pressure in the output HDF5 as still being in raw lattice
    units.

    NOTE on reference frames (bug found and fixed - see HANDOFF.md's
    "OPEN INVESTIGATION" for the full evidence/derivation): `Geometry/*`
    (positions, normals) is written once, frame-independent, and is
    expressed in the LRF (the rotor's own co-rotating frame - the mesh
    doesn't move in this frame, only the flow does). `Surface X/Y/Z-Force`
    as PowerFLOW stores it, however, is in the GLOBAL (lab, non-rotating)
    frame - confirmed empirically (a net in-plane wall-shear angle swept
    a full 360 deg once per revolution across a real 829-frame case,
    collapsing to a small residual once de-rotated by the file's own
    recorded angular rate) and independently corroborated from the file's
    own `start_time`/`lrf_constant_angular_vel_mag`/
    `lrf_initial_angular_rotation` metadata (an exact match, to three
    decimal degrees, against the empirically measured rate). to_h5()
    rotates `Surface X/Y/Z-Force` into the LRF (see `_rotation_angle()`)
    before writing, so every downstream consumer (FrictionLines,
    StripForces) sees vectors in the same frame as the geometry they're
    dotted against - no change needed on their end. `Skin Friction` and
    `Static Pressure` are stored as scalars in this format (confirmed via
    diagnose_snc.py), so they carry no frame ambiguity and are untouched.
    '''

    def __init__(self, filename: str):

        self.filename = filename
        # _LargeRecordNetcdfFile, not sio.netcdf_file directly - see its
        # own docstring: plain scipy corrupts the "measurements" record
        # size on a fine-enough DNS mesh (confirmed on a real 8 GB .snc
        # file), which this subclass fixes. Behaves identically to
        # sio.netcdf_file otherwise.
        self._f = _LargeRecordNetcdfFile(filename, mmap=False)
        self._decode_metadata()

    @staticmethod
    def _decode_packed(var) -> list:

        '''
        Split one of this format's flat, null-terminated char arrays
        (e.g. part_names, face_names) into a list of strings.
        '''

        raw = bytes(var[:].tobytes())
        return [s.decode('utf-8', errors='replace') for s in raw.split(b'\x00') if s]

    def _decode_metadata(self):

        '''
        Parse the small metadata arrays once at load time: variable
        names, part/face names and ids, rotor axis/angular velocity, and
        the lattice-unit scale table.
        '''

        f = self._f

        self.variable_names = self._decode_packed(f.variables['variable_long_names'])
        self.variable_index = {name: i for i, name in enumerate(self.variable_names)}
        self.variable_lattice_units = dict(zip(
            self.variable_names, self._decode_packed(f.variables['variable_lattice_unit_names'])
        ))

        self.part_names = self._decode_packed(f.variables['part_names'])
        self.face_names = self._decode_packed(f.variables['face_names'])
        self.face_ids = f.variables['face_ids'][:]

        self.lrf_axis_origin = np.array(f.variables['lrf_axis_origin'][:][0])
        self.lrf_axis_direction = np.array(f.variables['lrf_axis_direction'][:][0])
        self.lrf_angular_vel_lattice = float(f.variables['lrf_constant_angular_vel_mag'][:][0])

        # Needed to rotate Surface X/Y/Z-Force out of the global frame it's
        # stored in and into the LRF that Geometry/* (and therefore every
        # chordwise/spanwise/radial/tangential direction downstream) is
        # actually expressed in - see the class docstring's "NOTE on
        # reference frames" and HANDOFF.md's OPEN INVESTIGATION for the
        # full evidence/derivation. start_time/end_time are real per-frame
        # timestamps (lattice time units) - one entry per frame, same
        # indexing as `measurements`'s frame axis.
        self.lrf_has_constant_angular_vel = bool(f.variables['lrf_has_constant_angular_vel'][:][0])
        self.lrf_initial_angular_rotation = float(f.variables['lrf_initial_angular_rotation'][:][0])
        self.start_time = np.array(f.variables['start_time'][:], dtype=np.float64)
        self.end_time = np.array(f.variables['end_time'][:], dtype=np.float64)

        lx_names = self._decode_packed(f.variables['lx_names'])
        lx_scales = f.variables['lx_scales'][:]
        lx_offsets = f.variables['lx_offsets'][:]
        self.lattice_scales = dict(zip(lx_names, lx_scales))
        self.lattice_offsets = dict(zip(lx_names, lx_offsets))

        self.n_frames = f.variables['measurements'].shape[0]

    def _rotation_angle(self, frame: int, frame_meta: dict = None) -> float:

        '''
        Angle [rad] the LRF has rotated, at `frame`, relative to the
        fixed global frame `Surface X/Y/Z-Force` is stored in - see the
        class docstring's "NOTE on reference frames" and HANDOFF.md's
        OPEN INVESTIGATION for the full evidence/derivation.

        Two sources, in order of preference:

        1. `frame_meta['lrf_position_rad']`, if given - PowerFLOW's OWN
           authoritative angular position for this frame, from
           `exaritool nc-stats.ri <file>.snc -detail` (see
           parse_nc_stats()). Preferred when available: no sign/unit
           derivation needed, it's already the exact quantity this
           method exists to compute, straight from the tool that
           presumably computes it the same way internally.
        2. Otherwise, self-derived from this file's own recorded angular
           rate and per-frame timestamps (validated against an
           independent, empirically-measured drift rate - agreement to 3
           decimal degrees, HANDOFF.md's OPEN INVESTIGATION evidence #3):

               angle(frame) = lrf_initial_angular_rotation
                            + lrf_constant_angular_vel_mag * (start_time[frame] - start_time[0])

           `lrf_constant_angular_vel_mag` is already in radians per
           lattice time unit, and `start_time` in lattice time units, so
           this is directly in radians PROVIDED `LatticeTime`'s own scale
           factor is 1.0 (true for every case checked so far) - checked
           explicitly below rather than assumed, since this hasn't been
           validated for any case where it isn't.
        '''

        if frame_meta is not None and frame in frame_meta:
            return frame_meta[frame]['lrf_position_rad']

        if not self.lrf_has_constant_angular_vel:
            raise NotImplementedError(
                "lrf_has_constant_angular_vel is False for this file - the constant-rate "
                "rotation-angle formula above doesn't apply, and this case needs different "
                "handling (not yet implemented - see HANDOFF.md's OPEN INVESTIGATION)."
            )

        lattice_time_scale = self.lattice_scales.get('LatticeTime', 1.0)
        if not np.isclose(lattice_time_scale, 1.0):
            raise NotImplementedError(
                f"LatticeTime scale is {lattice_time_scale}, not 1.0 - the rotation-angle "
                "formula was only validated assuming this is exactly 1.0. Confirm the right "
                "conversion (multiply start_time by this scale?) before trusting the "
                "Surface_X/Y/Z-Force frame correction on this file."
            )

        return (self.lrf_initial_angular_rotation
                + self.lrf_angular_vel_lattice * (self.start_time[frame] - self.start_time[0]))

    @staticmethod
    def _rotate_about_axis(vectors: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:

        '''
        Rotate an (n, 3) array of vectors about a unit `axis` by `angle`
        [rad] (right-hand rule) - Rodrigues' rotation formula.
        '''

        axis = axis / np.linalg.norm(axis)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        cross = np.cross(axis, vectors)
        dot = vectors @ axis

        return vectors * cos_a + cross * sin_a + axis[None, :] * dot[:, None] * (1 - cos_a)

    def face_mask(self, name: str) -> np.ndarray:

        '''
        Boolean mask over all surfels, selecting those tagged with a face
        whose name contains `name` (e.g. 'Rotor_blade_1').

        Parameters
        ----------
        name : str
            Substring to match against face_names (e.g. '/Rotor::Rotor_blade_1'
            or just 'Rotor_blade_1').

        Returns
        -------
        np.ndarray of bool, shape (npoints,)
        '''

        matches = [fid for fid, fname in zip(self.face_ids, self.face_names) if name in fname]

        if not matches:
            raise ValueError(f"No face name matches '{name}'. Available: {self.face_names}")

        face = self._f.variables['face'][:]
        return np.isin(face, matches)

    def surfel_centroids(self) -> np.ndarray:

        '''
        Compute each surfel's centroid position (mean of its vertices),
        in lattice-length units.

        Returns
        -------
        np.ndarray, shape (npoints, 3)
        '''

        f = self._f
        first_vertex_refs = f.variables['first_vertex_refs'][:].astype(np.int64)
        vertex_refs = f.variables['vertex_refs'][:].astype(np.int64)
        vertex_coords = f.variables['vertex_coords'][:]

        npoints = len(first_vertex_refs)
        n_verts = np.diff(np.append(first_vertex_refs, len(vertex_refs)))

        surfel_index = np.repeat(np.arange(npoints), n_verts)
        pts = vertex_coords[vertex_refs]

        sums = np.column_stack([
            np.bincount(surfel_index, weights=pts[:, d], minlength=npoints)
            for d in range(3)
        ])

        return sums / n_verts[:, None]

    def surfel_normals(self) -> np.ndarray:
        return self._f.variables['surfel_normal'][:]

    def surfel_areas(self) -> np.ndarray:
        return self._f.variables['surfel_area'][:]

    def surface_split(self) -> np.ndarray:

        '''
        Boolean mask over all surfels, True for the "upper" surface: where
        the surfel normal has a positive component along the rotor's
        rotation axis (lrf_axis_direction).

        This is more robust than splitting by position (e.g. Y > 0),
        which becomes unreliable near the leading/trailing edges where
        thickness goes to zero - validated on this case: the normal-based
        ambiguous zone (|normal . axis| < 0.05) is ~45x narrower than the
        position-based one (|Y| < 0.0005), and where position-based is
        ambiguous the two methods agree only 55% of the time (~coin flip).

        Returns
        -------
        np.ndarray of bool, shape (npoints,)
        '''

        axis_direction = self.lrf_axis_direction / np.linalg.norm(self.lrf_axis_direction)
        return (self.surfel_normals() @ axis_direction) > 0

    def force_per_area_scale(self) -> float:

        '''
        Physical (Pa-equivalent) scale factor for ForcePerArea-class raw
        variables: Surface X/Y/Z-Force and Skin Friction. NOT valid for
        Static Pressure, which needs a different (Cp-based) translation -
        see the class docstring.
        '''

        return self.lattice_scales['LatticeDensity'] * self.lattice_scales['LatticeVelocity']**2

    def variable(self, name: str, frame: int = 0) -> np.ndarray:

        '''
        Raw (lattice-unit) values of one variable, for every surfel.

        Parameters
        ----------
        name : str
            One of self.variable_names (e.g. 'Static Pressure').
        frame : int
            Frame/timestep index, in [0, self.n_frames).
        '''

        idx = self.variable_index[name]
        return self._f.variables['measurements'][frame, idx, :]

    _FORCE_VARIABLE_NAMES = ['Surface X-Force', 'Surface Y-Force', 'Surface Z-Force']

    def _write_surfel_group(self, h5f, geo_path: str, data_path: str, mask: np.ndarray,
                             coords: np.ndarray, normals: np.ndarray, areas: np.ndarray,
                             force_scale: float, frame_meta: dict = None):

        '''
        Write one surfel selection's geometry (once, frame-independent -
        the raw .snc file's mesh/normals carry no frame axis, only
        measurements do) and variables (once per frame, stacked into a
        (n_frames, n_points) dataset) under the given Geometry/Data group
        paths. Shared by to_h5() for both the unsplit and
        upper/lower-split cases.

        Surface X/Y/Z-Force is handled specially: rotated about
        lrf_axis_direction, per-frame, from the global frame it's stored
        in into the LRF that Geometry/* is in - see the class docstring's
        "NOTE on reference frames" and _rotation_angle(). Everything else
        (Skin Friction, Static Pressure) is a scalar in this format, so
        it's written as-is, same as before.

        frame_meta : dict[int, dict], optional
            parse_nc_stats()'s return value (keyed by ABSOLUTE frame
            number, as used by exaritool - see _rotation_angle()), if
            available - passed straight through to _rotation_angle() per
            frame. Assumes this file's own row 0 corresponds to
            frame_meta's frame number 0; if this .snc is a partial dump
            starting at some other absolute frame, frame_meta's keys
            won't line up and every frame silently falls back to
            _rotation_angle()'s self-derived formula instead (still
            correct, just not cross-checked against PowerFLOW's own
            value) - not a crash, but worth being aware of.
        '''

        geo = h5f.create_group(geo_path)
        geo.create_dataset('X', data=coords[mask, 0])
        geo.create_dataset('Y', data=coords[mask, 1])
        geo.create_dataset('Z', data=coords[mask, 2])
        geo.create_dataset('Normal_X', data=normals[mask, 0])
        geo.create_dataset('Normal_Y', data=normals[mask, 1])
        geo.create_dataset('Normal_Z', data=normals[mask, 2])
        geo.create_dataset('Area', data=areas[mask])

        data = h5f.create_group(data_path)

        has_force = all(name in self.variable_index for name in self._FORCE_VARIABLE_NAMES)

        if has_force:

            if self.start_time is None and not frame_meta:
                raise ValueError(
                    f"'{self.filename}' has Surface X/Y/Z-Force but no start_time/"
                    "lrf_initial_angular_rotation/etc. metadata and no nc_stats_path was "
                    "given, so there's no way to compute the frame rotation these need "
                    "(see the class docstring's 'NOTE on reference frames') - refusing to "
                    "write potentially-wrong (still-global-frame) force data rather than "
                    "silently reproducing the bug this fixes."
                )

            axis_direction = self.lrf_axis_direction / np.linalg.norm(self.lrf_axis_direction)

            force_frames = []
            for frame in range(self.n_frames):
                raw = np.stack(
                    [self.variable(name, frame=frame)[mask] for name in self._FORCE_VARIABLE_NAMES],
                    axis=-1,
                )  # (n_selected, 3) - still global frame, raw lattice units
                angle = self._rotation_angle(frame, frame_meta=frame_meta)
                force_frames.append(self._rotate_about_axis(raw, axis_direction, -angle))

            force_all = np.stack(force_frames, axis=0) * force_scale  # (n_frames, n_selected, 3)

            for i, name in enumerate(self._FORCE_VARIABLE_NAMES):
                key = name.replace(' ', '_')
                dset = data.create_dataset(key, data=force_all[..., i])
                dset.attrs['lattice_unit_class'] = self.variable_lattice_units.get(name, '')
                dset.attrs['physical_units'] = True
                dset.attrs['rotated_to_lrf'] = True

        for name in self.variable_names:

            if has_force and name in self._FORCE_VARIABLE_NAMES:
                continue

            key = name.replace(' ', '_')
            unit_class = self.variable_lattice_units.get(name, '')
            is_physical = unit_class == 'LatticeForcePerArea'

            if not is_physical:
                warnings.warn(
                    f"'{name}' (unit class '{unit_class}') was written in raw lattice "
                    "units - no validated physical-unit conversion is applied here for "
                    "this unit class (e.g. Static Pressure needs PowerFLOW's Cp-based "
                    "translation; use pf2ens for that instead)."
                )

            values = np.stack(
                [self.variable(name, frame=frame)[mask] for frame in range(self.n_frames)],
                axis=0,
            )
            if is_physical:
                values = values * force_scale

            dset = data.create_dataset(key, data=values)
            dset.attrs['lattice_unit_class'] = unit_class
            dset.attrs['physical_units'] = is_physical

    def to_h5(self, output_path: str, face_name: str = None, surface_split: bool = False,
              nc_stats_path: str = None):

        '''
        Write coordinates (centroids), normals, areas, all variables (for
        every frame in the file), and rotor metadata to a plain HDF5
        file, optionally restricted to surfels matching face_name (e.g.
        one rotor blade).

        Positions, area and lrf_axis_origin are written in physical units
        (meters / m^2), scaled by the file's own LatticeLength (matching
        how Surface X/Y/Z-Force / Skin Friction are already scaled to Pa
        below) - NOT the raw lattice units surfel_centroids()/
        surfel_areas()/lrf_axis_origin return on their own.

        Each Data/<variable> dataset has shape (n_frames, n_points) -
        geometry (positions/normals/areas) is written once, since the
        raw .snc file carries no frame axis for it, only for
        measurements. Surface X/Y/Z-Force is additionally rotated,
        per-frame, from the global frame it's stored in into the LRF -
        see the class docstring's "NOTE on reference frames" and
        _rotation_angle().

        Parameters
        ----------
        output_path : str
        face_name : str, optional
            If given, only surfels matching this face (see face_mask) are
            written. Otherwise every surfel in the file is written.
        surface_split : bool, optional
            If True, split surfels into upper/lower surfaces (see
            surface_split()) and write them under separate
            Geometry/Upper, Geometry/Lower, Data/Upper, Data/Lower
            groups, instead of a single Geometry/Data pair. Off by
            default, by default False.
        nc_stats_path : str, optional
            Path to saved `exaritool nc-stats.ri <snc_path> -detail`
            output (see parse_nc_stats()) - same convention as
            converters.ensight_to_h5.convert_snc_to_h5()'s parameter of
            the same name. If given, its `lrf_position_rad` is used
            PREFERENTIALLY over _rotation_angle()'s self-derived formula
            (PowerFLOW's own authoritative angle - see _rotation_angle()),
            and its real per-frame `mid_s`/`start_ts`/`end_ts` are written
            into Metadata alongside `lrf_position_rad`, matching
            EnsightSeriesWriter's schema for the pressure branch - so
            SurfaceVariable.timetrace()/periodogram() (which already look
            for Metadata/mid_s) get a real sampling rate here too. If
            omitted, rotation falls back to the self-derived formula, and
            Metadata gets this file's own raw `start_time`/`end_time`
            instead (lattice time units, NOT seconds - no validated
            conversion to physical time exists without nc_stats_path;
            written for reference only, not as a `mid_s`-equivalent).
        '''

        length_scale = self.lattice_scales['LatticeLength']
        coords = self.surfel_centroids() * length_scale
        normals = self.surfel_normals()
        areas = self.surfel_areas() * length_scale ** 2

        base_mask = self.face_mask(face_name) if face_name is not None else np.ones(len(areas), dtype=bool)

        frame_meta = parse_nc_stats(nc_stats_path) if nc_stats_path else None

        with h5py.File(output_path, 'w') as h5f:

            meta = h5f.create_group('Metadata')
            meta.create_dataset('lrf_axis_origin', data=self.lrf_axis_origin * length_scale)
            meta.create_dataset('lrf_axis_direction', data=self.lrf_axis_direction)
            meta.create_dataset('frame_index', data=np.arange(self.n_frames))
            meta.attrs['lrf_angular_vel_lattice'] = self.lrf_angular_vel_lattice

            if frame_meta:
                # Real, PowerFLOW-computed per-frame timing/angle - same
                # schema as converters.ensight_to_h5.EnsightSeriesWriter,
                # so SurfaceVariable's existing Metadata/mid_s lookup
                # (timetrace()/periodogram()) works here too. NaN for any
                # frame missing from frame_meta (see _write_surfel_group()'s
                # docstring on frame-index alignment).
                mid_s = np.array([frame_meta.get(i, {}).get('mid_s', np.nan) for i in range(self.n_frames)])
                lrf_position_rad = np.array(
                    [frame_meta.get(i, {}).get('lrf_position_rad', np.nan) for i in range(self.n_frames)]
                )
                meta.create_dataset('mid_s', data=mid_s)
                meta.create_dataset('lrf_position_rad', data=lrf_position_rad)
            elif self.start_time is not None:
                # No nc_stats_path - this file's own raw timestamps, for
                # reference only. NOT seconds (no validated conversion -
                # see class docstring) - deliberately NOT called mid_s,
                # unlike the branch above, so nothing downstream mistakes
                # this for a real sampling rate.
                meta.create_dataset('start_time_lattice', data=self.start_time)
                meta.create_dataset('end_time_lattice', data=self.end_time)

            for k, v in self.lattice_scales.items():
                meta.attrs[f'scale_{k}'] = v
            for k, v in self.lattice_offsets.items():
                meta.attrs[f'offset_{k}'] = v

            force_scale = self.force_per_area_scale()

            if surface_split:
                upper = self.surface_split()
                groups = {'Upper': base_mask & upper, 'Lower': base_mask & ~upper}
            else:
                groups = {None: base_mask}

            for label, mask in groups.items():
                geo_path = f'Geometry/{label}' if label else 'Geometry'
                data_path = f'Data/{label}' if label else 'Data'
                self._write_surfel_group(h5f, geo_path, data_path, mask, coords, normals, areas,
                                          force_scale, frame_meta=frame_meta)

    def close(self):
        self._f.close()
