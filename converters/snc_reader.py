import warnings
from functools import reduce
from operator import mul
import h5py
import numpy as np
import scipy.io as sio
import scipy.io._netcdf as _nc


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

    Fix: never trust the file's own `vsize` field when accumulating
    `self._recsize` - recompute the true value ourselves from the
    variable's own (correctly parsed, NOT subject to this 32-bit limit)
    shape/dtype instead, using the exact formula scipy's own comment
    documents (product of the non-record dimensions x itemsize, rounded
    up to the next multiple of 4). Everything else below is otherwise
    IDENTICAL to scipy.io._netcdf.netcdf_file._read_var_array - there is
    no smaller public hook to override, the fix has to happen inside
    this exact loop, so this necessarily duplicates scipy's private
    method rather than patching one line of it. Tied to scipy's private
    API by nature - if a scipy upgrade changes this method's internals,
    this override needs revisiting (it will fail loudly with an
    AttributeError pointing at whatever's missing, not silently misread
    data, since it calls straight through to the same private helpers
    scipy's own unmodified method does).
    '''

    def _read_var_array(self):

        header = self.fp.read(4)
        if header not in (_nc.ZERO, _nc.NC_VARIABLE):
            raise ValueError("Unexpected header.")

        begin = 0
        dtypes = {'names': [], 'formats': []}
        rec_vars = []
        count = self._unpack_int()

        for i in range(count):

            (name, dimensions, shape, attributes,
             typecode, size, dtype_, begin_, vsize) = self._read_var()

            if shape and shape[0] is None:  # record variable

                rec_vars.append(name)

                # NOT `vsize` (the file's own, possibly-corrupted field -
                # see class docstring) - recomputed independently from
                # shape/dtype instead.
                true_vsize = reduce(mul, shape[1:], 1) * size
                true_vsize = ((true_vsize + 3) // 4) * 4
                self.__dict__['_recsize'] += true_vsize

                if begin == 0:
                    begin = begin_
                dtypes['names'].append(name)
                dtypes['formats'].append(str(shape[1:]) + dtype_)

                if typecode in 'bch':
                    actual_size = reduce(mul, (1,) + shape[1:]) * size
                    padding = -actual_size % 4
                    if padding:
                        dtypes['names'].append('_padding_%d' % i)
                        dtypes['formats'].append('(%d,)>b' % padding)

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

            if len(rec_vars) == 1:
                dtypes['names'] = dtypes['names'][:1]
                dtypes['formats'] = dtypes['formats'][:1]

            if self.use_mmap:
                rec_array = self._mm_buf[begin:begin + self._recs * self._recsize].view(dtype=dtypes)
                rec_array.shape = (self._recs,)
            else:
                pos = self.fp.tell()
                self.fp.seek(begin)
                rec_array = _nc.frombuffer(self.fp.read(self._recs * self._recsize), dtype=dtypes).copy()
                rec_array.shape = (self._recs,)
                self.fp.seek(pos)

            for var in rec_vars:
                self.variables[var].__dict__['data'] = rec_array[var]


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

        lx_names = self._decode_packed(f.variables['lx_names'])
        lx_scales = f.variables['lx_scales'][:]
        lx_offsets = f.variables['lx_offsets'][:]
        self.lattice_scales = dict(zip(lx_names, lx_scales))
        self.lattice_offsets = dict(zip(lx_names, lx_offsets))

        self.n_frames = f.variables['measurements'].shape[0]

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

    def _write_surfel_group(self, h5f, geo_path: str, data_path: str, mask: np.ndarray,
                             coords: np.ndarray, normals: np.ndarray, areas: np.ndarray,
                             force_scale: float):

        '''
        Write one surfel selection's geometry (once, frame-independent -
        the raw .snc file's mesh/normals carry no frame axis, only
        measurements do) and variables (once per frame, stacked into a
        (n_frames, n_points) dataset) under the given Geometry/Data group
        paths. Shared by to_h5() for both the unsplit and
        upper/lower-split cases.
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

        for name in self.variable_names:
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

    def to_h5(self, output_path: str, face_name: str = None, surface_split: bool = False):

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
        measurements.

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
        '''

        length_scale = self.lattice_scales['LatticeLength']
        coords = self.surfel_centroids() * length_scale
        normals = self.surfel_normals()
        areas = self.surfel_areas() * length_scale ** 2

        base_mask = self.face_mask(face_name) if face_name is not None else np.ones(len(areas), dtype=bool)

        with h5py.File(output_path, 'w') as h5f:

            meta = h5f.create_group('Metadata')
            meta.create_dataset('lrf_axis_origin', data=self.lrf_axis_origin * length_scale)
            meta.create_dataset('lrf_axis_direction', data=self.lrf_axis_direction)
            meta.create_dataset('frame_index', data=np.arange(self.n_frames))
            meta.attrs['lrf_angular_vel_lattice'] = self.lrf_angular_vel_lattice
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
                self._write_surfel_group(h5f, geo_path, data_path, mask, coords, normals, areas, force_scale)

    def close(self):
        self._f.close()
