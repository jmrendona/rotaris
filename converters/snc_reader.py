import warnings
import h5py
import numpy as np
import scipy.io as sio


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

    NOTE on units: variable()/wall_shear-style raw access returns values
    exactly as stored in the file (PowerFLOW "lattice units"). to_h5()
    converts ForcePerArea-class variables (Surface X/Y/Z-Force, Skin
    Friction) to physical units (Pa) using force_per_area_scale(),
    validated against this file's own stored Skin Friction (correlation
    ~0.999). Static Pressure is NOT converted - it needs PowerFLOW's
    internal Cp-based translation (see the PowerVIZ User's Guide, Appendix
    B), which isn't implemented here; use pf2ens for physical-unit
    pressure instead, and treat Data/Static_Pressure in the output HDF5 as
    still being in raw lattice units.
    '''

    def __init__(self, filename: str):

        self.filename = filename
        self._f = sio.netcdf_file(filename, mmap=False)
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
        geo.create_dataset('NX', data=normals[mask, 0])
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

        coords = self.surfel_centroids()
        normals = self.surfel_normals()
        areas = self.surfel_areas()

        base_mask = self.face_mask(face_name) if face_name is not None else np.ones(len(areas), dtype=bool)

        with h5py.File(output_path, 'w') as h5f:

            meta = h5f.create_group('Metadata')
            meta.create_dataset('lrf_axis_origin', data=self.lrf_axis_origin)
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
