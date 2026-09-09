import numpy as np
import h5py


def cylindrical_velocity(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, angle_deg: float,
                          axis: str = 'z', r: np.ndarray = None, omega_rad_s: float = None):

    '''
    Convert a Cartesian velocity sample (vx, vy, vz) - taken on a ONE-
    SIDED meridional plane at a single fixed azimuth `angle_deg` (see
    converters.fnc_plane.meridional_plane_points with a one-sided
    inplane_range=(0, R) - every point on such a plane shares exactly
    ONE azimuth, since the whole rigid point set is built at angle=0 and
    then rotated as one unit) - into cylindrical components (v_r,
    v_theta, v_z), in the rotor's own rotating frame.

    This is TWO SEPARATE things, both needed, not one:

    1. A pure COORDINATE rotation: v_r/v_theta are (vx, vy) projected
       onto the LOCAL radial/tangential basis at this plane's azimuth,
       (cos/sin(angle_deg), -sin/cos(angle_deg)) - trivial and exact,
       needs nothing beyond angle_deg. This is what runs unconditionally
       below.
    2. An ADDITIVE frame correction, v_theta_relative = v_theta_absolute
       - omega_rad_s * r: needed ONLY IF pf2ens's vx/vy/vz are the
         ABSOLUTE (lab-frame) velocity rather than already the RELATIVE
         (rotating-frame) velocity. **This has NOT been verified for
         this project** - nothing in this codebase has checked which one
         pf2ens actually exports (this is a structurally identical
         question to the Surface_X/Y/Z-Force global-vs-LRF bug found and
         fixed in converters/snc_reader.py, but for a completely
         different tool/extraction path - it needs its own, separate
         empirical check, not an assumption carried over from that fix).
         Confirmed correct in this function ONLY as a piece of math (see
         "Validated" below, a pure solid-body-rotation test case) - NOT
         validated against real pf2ens output, since that requires
         knowing the answer to the open question above first. Leave
         `omega_rad_s=None` (default) to skip this step entirely.

    Parameters
    ----------
    vx, vy, vz : np.ndarray, same shape
        Cartesian velocity components, in whatever frame pf2ens exports
        (see point 2 above - this function does not know or assume
        which one).
    angle_deg : float
        This plane's azimuth (degrees) - e.g. Geometry.attrs['angle_deg']
        from an extract_to_h5() meridional output.
    axis : str
        Rotor rotation axis ('x', 'y', or 'z') - same convention as
        converters.fnc_plane.meridional_plane_points.
    r : np.ndarray, optional
        Physical radius [m] from the rotation axis, one per sample -
        REQUIRED if omega_rad_s is given, unused otherwise. Compute the
        same way RotorBladePosition does (rel = point - axis_origin;
        r = hypot of the two non-axis components) - see this module's
        docstring note on axis_origin/case-origin alignment, another
        open item shared with RotorBladePosition, not re-solved here.
    omega_rad_s : float, optional
        SIGNED angular velocity [rad/s] of the LRF, in the SAME azimuth
        sign convention as `angle_deg` (positive/negative exactly as
        this dataset's own azimuth increases/decreases with time - e.g.
        `omega = rpm * 2*pi/60`, sign-matched to the actual measured
        rotation direction, not assumed). If given, applies the frame
        correction above; if None (default), v_theta is left as whatever
        frame vx/vy/vz were already in.

    Returns
    -------
    v_r, v_theta, v_z : np.ndarray, same shape as vx/vy/vz

    Validated (this function's pure math, not real pf2ens data):
    - Coordinate rotation checked at angle_deg = 0, 90, 180, 270 against
      hand-computed expected (v_r, v_theta) for known (vx, vy) - exact
      match.
    - Frame correction checked against an analytic solid-body-rotation
      field (v_absolute = omega x r, i.e. v_theta_absolute = omega*r,
      v_r = v_z = 0 everywhere) - recovers v_theta_relative = 0 exactly,
      as it must (a rigid rotation has zero velocity relative to its own
      rotating frame).
    '''

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if axis_idx == 2:  # z is the rotation axis - (vx, vy) are in-plane, vz is axial
        v_in1, v_in2, v_ax = vx, vy, vz
    elif axis_idx == 1:  # y is the rotation axis
        v_in1, v_in2, v_ax = vx, -vz, vy
    else:  # x is the rotation axis
        v_in1, v_in2, v_ax = vy, vz, vx

    v_r = v_in1 * cos_t + v_in2 * sin_t
    v_theta = -v_in1 * sin_t + v_in2 * cos_t
    v_z = v_ax

    if omega_rad_s is not None:
        if r is None:
            raise ValueError("r is required when omega_rad_s is given (needed for omega*r).")
        v_theta = v_theta - omega_rad_s * r

    return v_r, v_theta, v_z


class TipVortexPhaseAverage:

    '''
    Phase-locked, wake-age-referenced averaging across a set of ONE-
    SIDED meridional-plane extractions (converters.fnc_plane.extract_to_h5,
    called once per azimuth with inplane_range=(0, R) - NOT the two-sided
    default), following Quaglia et al.'s tip-vortex tracking methodology:
    N planes evenly spaced by `spacing_deg`, spanning the FULL 360
    degrees (one-sided planes don't get the other half for free the way
    a two-sided diametral plane does - see this class's design
    discussion), with `spacing_deg` chosen to exactly match this
    dataset's own per-frame rotor rotation angle.

    The core idea: a wake structure convects azimuthally at
    (approximately) the rotor's own rotation rate - the standard
    classical "wake age" assumption in rotor aerodynamics
    (psi_w = Omega*(t - t_shed)). If the plane spacing exactly equals the
    per-frame rotation angle, then relabeling each physical plane's index
    by exactly -1 (mod n_planes) every frame reconstructs a "virtual",
    co-rotating sequence of planes that always shows the SAME relative
    wake age, regardless of absolute time or absolute azimuth. Averaging
    all samples sharing one such label across every available frame
    gives a phase-locked average at that one age - equivalent to
    referencing the average to "time since shedding" without needing to
    explicitly detect any shedding event (see age_label()'s docstring
    for the exact index formula and why no shedding-detection step is
    needed).

    For THIS project's rotor (2 blades), a blade passes any fixed
    azimuth every 180 degrees - exactly half of one full 360-degree
    relabeling cycle - so label k and label k + n_planes//2 both
    correspond to "the same relative age since the most recent blade
    passage", just referencing different blades. Under blade-to-blade
    symmetry (not assumed or exploited here - left as a documented
    possibility) these could later be pooled for roughly double the
    effective sample count per age bin.

    Parameters
    ----------
    h5_paths : list of str
        One extract_to_h5() output per azimuth, each built with a ONE-
        SIDED inplane_range and geometry_attrs={'kind': 'meridional',
        'angle_deg': <this plane's azimuth>, 'axis': ...} (see
        converters.fnc_plane.extract_to_h5's geometry_attrs parameter).
        All files must share the same grid (same points, same order) and
        the same frame range/count.
    spacing_deg : float
        Angular spacing between consecutive planes, in azimuth degrees -
        MUST equal this dataset's actual per-frame rotation angle for
        the relabeling to be exact (not approximate/interpolated). For
        this project's 6e-5-6000rpm case, that's the same ~2.0011
        deg/frame established independently in the Surface_Force
        reference-frame investigation (see HANDOFF.md) - use that same
        source of truth, don't re-derive it separately and risk a
        mismatch.
    direction_sign : int, optional
        +1 or -1: does azimuth increase (+1) or decrease (-1) as frame
        index increases, in the SAME sign convention `angle_deg` values
        are given in? If omitted, inferred automatically from the sign
        of Metadata/lrf_position_rad's own frame-to-frame trend in the
        first file (if present - i.e. that file was extracted with
        nc_stats_path) - preferred over guessing, since it reuses this
        exact dataset's own recorded rotation, not an assumption carried
        over from a different file/case. Raises if omitted and
        Metadata/lrf_position_rad isn't available.
    '''

    def __init__(self, h5_paths: list, spacing_deg: float, direction_sign: int = None):

        if len(h5_paths) < 2:
            raise ValueError(f"Need at least 2 plane files, got {len(h5_paths)}.")

        entries = []
        for path in h5_paths:
            with h5py.File(path, 'r') as f:
                if f['Geometry'].attrs.get('kind') != 'meridional':
                    raise ValueError(f"'{path}': Geometry.attrs['kind'] is not 'meridional'.")
                angle_deg = float(f['Geometry'].attrs['angle_deg'])
                n_points = f['Geometry/X'].shape[0]
                n_frames = f['Metadata/frame_index'].shape[0]
            entries.append({'path': path, 'angle_deg': angle_deg, 'n_points': n_points, 'n_frames': n_frames})

        entries.sort(key=lambda e: e['angle_deg'])

        angles = np.array([e['angle_deg'] for e in entries])
        expected = (angles[0] + spacing_deg * np.arange(len(entries))) % 360
        actual = angles % 360
        if not np.allclose(expected, actual, atol=1e-6):
            raise ValueError(
                f"Plane angles {angles.tolist()} are not evenly spaced by spacing_deg={spacing_deg} "
                "starting from the lowest angle - check h5_paths covers a consistent sweep."
            )

        n_points_set = {e['n_points'] for e in entries}
        n_frames_set = {e['n_frames'] for e in entries}
        if len(n_points_set) != 1:
            raise ValueError(f"Plane files have inconsistent point counts: {n_points_set}")
        if len(n_frames_set) != 1:
            raise ValueError(f"Plane files have inconsistent frame counts: {n_frames_set}")

        self.entries = entries
        self.spacing_deg = spacing_deg
        self.n_planes = len(entries)
        self.n_points = n_points_set.pop()
        self.n_frames = n_frames_set.pop()

        if direction_sign is None:
            direction_sign = self._infer_direction_sign(entries[0]['path'])
        if direction_sign not in (1, -1):
            raise ValueError(f"direction_sign must be 1 or -1, got {direction_sign!r}")
        self.direction_sign = direction_sign

        self._load_shared_grid(entries[0])

    def _load_shared_grid(self, entry: dict):

        '''
        Reconstruct the local (in-plane, axial) grid coordinates ONE
        plane's own points un-rotate to - every plane shares this SAME
        local template (meridional_plane_points() builds one rectangle
        at angle=0, then rotates the WHOLE rigid point set to each
        plane's own azimuth), so it applies equally to plot_age_label()'s
        phase-averaged data, which is indexed by the same per-point grid
        position regardless of which physical plane/frame contributed to
        a given age label. Only used for plotting - compute() doesn't
        need this at all.

        Reuses fnc_plane's own _rotate_about_axis() (exactly what
        plot_frame() un-rotates with) rather than re-deriving the
        per-axis rotation by hand - that hand-derivation was tried here
        first and got the y-axis case wrong (confirmed by differentiating
        _rotate_about_axis's own forward rotation numerically - the y
        rotation isn't a simple cyclic relabeling of the x/z-axis
        pattern, unlike x and z, which are each other's mirror). Reusing
        the one proven-correct implementation instead of a second,
        independently-derived one avoids that whole class of mistake.
        '''

        from converters.fnc_plane import _rotate_about_axis

        with h5py.File(entry['path'], 'r') as f:
            geo = f['Geometry']
            self.axis = str(geo.attrs['axis'])
            self.grid_shape = tuple(int(n) for n in geo.attrs['grid_shape']) if 'grid_shape' in geo.attrs else None
            points = np.stack([geo['X'][:], geo['Y'][:], geo['Z'][:]], axis=1).astype(float)

        axis_idx = {'x': 0, 'y': 1, 'z': 2}[self.axis.lower()]
        inplane_idx = [i for i in range(3) if i != axis_idx][0]
        local = _rotate_about_axis(points, self.axis, -entry['angle_deg'])

        # local_radius is the physical radius for a ONE-SIDED plane (starts at
        # the rotation axis) - kept as a flat (n_points,) array, reshaped per-plot.
        self.local_radius = local[:, inplane_idx]
        self.local_axial = local[:, axis_idx]

    @staticmethod
    def _infer_direction_sign(path: str) -> int:

        '''
        +1/-1 from the sign of Metadata/lrf_position_rad's own frame-to-
        frame trend in one plane file - see class docstring's
        direction_sign parameter for why this is preferred over a
        hardcoded/assumed sign.
        '''

        with h5py.File(path, 'r') as f:
            if 'Metadata/lrf_position_rad' not in f:
                raise ValueError(
                    f"'{path}' has no Metadata/lrf_position_rad (extracted without nc_stats_path) - "
                    "direction_sign can't be inferred automatically; pass it explicitly."
                )
            lrf_position_rad = f['Metadata/lrf_position_rad'][:]

        diffs = np.diff(lrf_position_rad)
        if np.all(np.isnan(diffs)):
            raise ValueError(f"'{path}' Metadata/lrf_position_rad is all-NaN - can't infer direction_sign.")

        mean_diff = np.nanmean(diffs)
        if mean_diff == 0:
            raise ValueError(f"'{path}' Metadata/lrf_position_rad doesn't change across frames.")

        return 1 if mean_diff > 0 else -1

    def age_label(self, plane_index: int, frame_index: int) -> int:

        '''
        The relative "wake age" label (0 to n_planes-1) that plane
        `plane_index` (0-indexed into the angle-sorted h5_paths) holds
        at `frame_index`.

            age_label(p, f) = (p - direction_sign * f) mod n_planes

        Confirmed against the exact worked example this design was built
        from: with direction_sign=+1, at f=0 every plane's own index IS
        its label (age_label(p, 0) == p); at f=1 (one plane-spacing of
        rotation later), plane 1 holds label 0 and plane 0 wraps to label
        n_planes-1 - i.e. "what was plane 1 is now label 0, what was
        plane 0 becomes label n".
        '''

        return (plane_index - self.direction_sign * frame_index) % self.n_planes

    def compute(self, variables: list, cylindrical: bool = True, velocity_keys=('vx', 'vy', 'vz'),
                omega_rad_s: float = None, axis: str = 'z', axis_origin=None):

        '''
        Stream through every plane file once (never holding more than one
        plane's raw Data in memory at a time - only the running per-age-
        label sums/counts, sized n_planes x n_points, persist throughout),
        and return the phase-locked average of `variables` at every wake-
        age label.

        Parameters
        ----------
        variables : list of str
            Data/<name> keys to average, from each plane file (e.g.
            ['vx', 'vy', 'vz'] to get cylindrical velocity out, or e.g.
            ['p'] for pressure - any mix is fine; only the ones matching
            `velocity_keys` get the cylindrical treatment below).
        cylindrical : bool
            If True (default) and all three of `velocity_keys` are
            present in `variables`, replace them with ('v_r', 'v_theta',
            'v_z') in the output, converted via cylindrical_velocity()
            (per-plane, using that plane's own angle_deg - see that
            function's docstring for the coordinate-rotation-only vs.
            +frame-correction distinction). Any other requested variable
            is averaged as a plain scalar, untouched.
        omega_rad_s : float, optional
            Forwarded to cylindrical_velocity() - see its docstring. None
            (default) skips the (unverified-for-this-tool) frame
            correction and leaves v_theta in whatever frame vx/vy/vz
            were already in.
        axis_origin : np.ndarray, shape (3,), optional
            Required only if omega_rad_s is given (needs radius per
            point) - in the SAME coordinate frame as this plane file's
            Geometry/X,Y,Z (see this module's docstring note on
            axis_origin/case-origin alignment - matches
            RotorBladePosition's own, not-independently-re-verified,
            choice of using the raw, uncorrected lrf_axis_origin).

        Returns
        -------
        dict
            {age_label: {var_name: np.ndarray (n_points,), ...,
                         'count': np.ndarray (n_points,)}}
            for every age_label in range(n_planes). 'count' is the
            number of valid (plane, frame) samples that contributed to
            each point - varies per point since Data/valid is per-point,
            per-frame (points can be measurement-volume-invalid on some
            frames and not others).
        '''

        want_cylindrical = cylindrical and all(v in variables for v in velocity_keys)
        out_names = (
            [v for v in variables if v not in velocity_keys] + ['v_r', 'v_theta', 'v_z']
            if want_cylindrical else list(variables)
        )

        sums = {name: np.zeros((self.n_planes, self.n_points)) for name in out_names}
        counts = {name: np.zeros((self.n_planes, self.n_points)) for name in out_names}

        r = None
        if omega_rad_s is not None:
            if axis_origin is None:
                raise ValueError("axis_origin is required when omega_rad_s is given.")
            axis_origin = np.asarray(axis_origin, dtype=float)

        for plane_index, entry in enumerate(self.entries):

            with h5py.File(entry['path'], 'r') as f:

                valid = f['Data/valid'][:]  # (n_frames, n_points)
                raw = {name: f[f'Data/{name}'][:] for name in variables}  # each (n_frames, n_points)

                if omega_rad_s is not None:
                    points = np.stack([f['Geometry/X'][:], f['Geometry/Y'][:], f['Geometry/Z'][:]], axis=1)
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
                    other_idx = [i for i in range(3) if i != axis_idx]
                    rel = points - axis_origin
                    r = np.hypot(rel[:, other_idx[0]], rel[:, other_idx[1]])

            if want_cylindrical:
                v_r, v_theta, v_z = cylindrical_velocity(
                    raw[velocity_keys[0]], raw[velocity_keys[1]], raw[velocity_keys[2]],
                    entry['angle_deg'], axis=axis, r=r, omega_rad_s=omega_rad_s,
                )
                frame_values = {name: raw[name] for name in variables if name not in velocity_keys}
                frame_values['v_r'], frame_values['v_theta'], frame_values['v_z'] = v_r, v_theta, v_z
            else:
                frame_values = raw

            for frame_index in range(self.n_frames):

                label = self.age_label(plane_index, frame_index)
                mask = valid[frame_index]

                for name in out_names:
                    values = frame_values[name][frame_index]
                    sums[name][label] += np.where(mask, values, 0.0)
                    counts[name][label] += mask

        result = {}
        for label in range(self.n_planes):
            entry = {}
            for name in out_names:
                with np.errstate(invalid='ignore', divide='ignore'):
                    entry[name] = np.where(counts[name][label] > 0, sums[name][label] / counts[name][label], np.nan)
            entry['count'] = counts[out_names[0]][label]
            result[label] = entry

        return result

    def plot_age_label(self, result: dict, label: int, variable: str, min_count: int = 1,
                        ax=None, cmap: str = 'turbo', levels=100, cbar_label: str = None,
                        savepath: str = None, dpi: int = 150):

        '''
        Filled-contour plot of one age label's phase-locked-averaged
        field, on this pipeline's native (radius, axial) grid - same
        style as converters.fnc_plane.plot_frame(), but showing whatever
        radial range the input planes actually cover (one-sided, e.g.
        [0, R] - NOT mirrored into a full [-R, R] diameter the way a
        two-sided extraction's quick-look plot would be. If you want a
        full-diameter picture instead, that needs deliberately pairing
        this label with another one - not done here, see this class's
        design discussion for why that's a separate decision, not a
        default).

        Parameters
        ----------
        result : dict
            compute()'s return value.
        label : int
            Which age label (0 to n_planes-1) to plot.
        variable : str
            Which key from result[label] to plot (e.g. 'v_r', 'v_theta',
            'v_z', or any other variable compute() was asked for).
        min_count : int
            Points with fewer than this many contributing (plane, frame)
            samples are masked as NaN rather than plotted - guards
            against a near-empty average (e.g. a point valid in only one
            or two frames) looking as trustworthy as a well-sampled one.
        ax, cmap, levels, savepath, dpi : see plot_frame() - same
            conventions.
        cbar_label : str, optional
            Colorbar label (default: `variable` itself).

        Returns
        -------
        (fig, ax)
        '''

        if self.grid_shape is None:
            raise ValueError(
                "No Geometry/grid_shape on the input plane files - plot_age_label() only works for "
                "a 2D grid (meridional_plane_points' output), not an arbitrary point cloud."
            )
        if label not in result:
            raise ValueError(f"label={label} not in result (available: {sorted(result.keys())})")
        if variable not in result[label]:
            raise ValueError(
                f"'{variable}' not in result[{label}] (available: "
                f"{[k for k in result[label] if k != 'count']})"
            )

        import matplotlib.pyplot as plt

        u = self.local_radius.reshape(self.grid_shape)
        a = self.local_axial.reshape(self.grid_shape)
        values = result[label][variable].reshape(self.grid_shape).astype(float).copy()
        count = result[label]['count'].reshape(self.grid_shape)
        values[count < min_count] = np.nan

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        contour = ax.contourf(u, a, values, levels=levels, cmap=cmap, extend='both')
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(cbar_label or variable)
        ax.set_xlabel('radius [m]')
        ax.set_ylabel('axial [m]')
        ax.set_title(f'{variable}, age label {label} (of {self.n_planes})')
        ax.set_aspect('equal')

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax
