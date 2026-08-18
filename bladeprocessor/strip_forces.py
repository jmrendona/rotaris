import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18
})


class StripForces:

    '''
    Per-radial-strip (optionally per-radial-AND-chordwise-strip) sectional
    aerodynamic loading, resolved in time - the raw input Hanson's tonal
    noise method needs (harmonics of unsteady sectional loading), computed
    directly from a converters.snc_reader.SNCReader.to_h5() conversion of
    the raw .snc file.

    Replaces a manual, per-frame PowerVIZ "Force Graph" CSV export (see
    converters/forces_strip.py's ForcesCSVConverter, the previous path
    this is meant to replace) with a direct read of the same raw surfel
    force data this project's other tools already use - no PowerVIZ
    session, no per-case manual export, and no statistical guessing of
    which axis is which (see ForcesCSVConverter._detect_component_mapping()'s
    mean/std heuristic - fragile, since it INFERS axial/tangential/radial
    from the data itself rather than from known geometry). Here,
    axial/tangential/radial are derived directly from the rotor's own
    rotation axis (Metadata/lrf_axis_origin, lrf_axis_direction) - no
    guessing involved.

    Per-surfel force vector = Surface_X/Y/Z-Force (Pa, already converted
    to physical units by SNCReader.to_h5() - see its own docstring) times
    Area (m²) - matches how converters.forces_strip.ForcesCSVConverter's
    PowerVIZ-exported "PerSegment...Force[Force:newton]" columns are
    already actual forces (Newtons), not per-area tractions.

    Physical basis, per surfel:
        axial      = the rotor's rotation axis direction (constant)
        radial     = unit vector from the axis to the surfel, in the
                     plane perpendicular to the axis (points AWAY from
                     the axis - unambiguous, no sign convention to check)
        tangential = axial x radial (a right-handed basis) - whether
                     this points in the direction of rotation or against
                     it depends on which way axis_direction happens to
                     point in this file, same kind of arbitrary-sign
                     issue as everywhere else in this project needing a
                     check (see reverse_chord/flip_direction elsewhere) -
                     use flip_tangential if it comes out backwards.
        Force is projected onto this basis by a dot product at every
        surfel, every frame, THEN summed within each strip - not the
        other way around (summing raw X/Y/Z force first, then somehow
        rotating a bulk vector, would be wrong: the basis itself varies
        across the strip's own surfels, since each one has its own
        radial direction).
    axial_unit's own sign (does "positive axial" mean thrust up or down)
    is likewise whatever this file's own lrf_axis_direction happens to
    be - use flip_axial if it comes out with the wrong sign for what you
    expect physically.

    Chordwise (non-compactness) subdivision: at high enough frequency
    (Hanson's method's tonal harmonics, or just a high blade-passing
    frequency to begin with), a single strip's ENTIRE chord can no
    longer be treated as one compact source - the acoustic phase varies
    measurably across it. compute()'s n_chord_bins subdivides each
    radial strip further, along the raw chord axis, into that many
    additional sub-strips, each independently time-resolved - off by
    default (n_chord_bins=None), matching the original PowerVIZ-based
    per-strip-only approach (converters/forces_strip.py), which never
    subdivided the chord.

    Parameters
    ----------
    filename : str
        Path to a SNCReader.to_h5() HDF5 file - either the unsplit
        branch (a single Geometry/Data group) or the surface_split=True
        branch (Geometry/Data under Upper AND Lower) - both work, and if
        split, Upper+Lower are combined before summing (this class needs
        the TOTAL force per strip, not one surface's contribution).
    r_tip : float, optional
        Physical tip radius [m], for reference only - not required for
        anything below to run.
    rpm : float, optional
        Rotor speed [rev/min] - needed for phase_lock() and harmonics()
        (converting frame index -> azimuth angle / rotation frequency).
        Not required for compute()/total_loads()/plot_bar_forces().
    span_axis, chord_axis, thickness_axis : int
        Which raw position column (0=X, 1=Y, 2=Z) is spanwise, chordwise,
        thickness-wise - see FrictionLines'/SurfaceVariable's docstrings
        for the same parameters; same defaults (0, 2, 1).
    '''

    def __init__(self, filename: str, r_tip: float = None, rpm: float = None, span_axis: int = 0,
                 chord_axis: int = 2, thickness_axis: int = 1):

        self.filename = filename
        self.r_tip = r_tip
        self.rpm = rpm
        self.span_axis = span_axis
        self.chord_axis = chord_axis
        self.thickness_axis = thickness_axis
        self._load()

    def _load(self):

        '''Load geometry, areas, per-frame force field, and rotor axis metadata - combining Upper+Lower if split.'''

        with h5py.File(self.filename, 'r') as f:

            axis_origin = f['Metadata/lrf_axis_origin'][:]
            axis_direction = f['Metadata/lrf_axis_direction'][:]
            self.axis_origin = axis_origin
            self.axis_direction = axis_direction / np.linalg.norm(axis_direction)
            self.n_frames = f['Metadata/frame_index'].shape[0]

            labels = ['Upper', 'Lower'] if 'Upper' in f['Geometry'] else [None]

            positions_parts, area_parts, force_parts = [], [], []

            for label in labels:

                geo_path = f'Geometry/{label}' if label else 'Geometry'
                data_path = f'Data/{label}' if label else 'Data'
                geo = f[geo_path]
                data = f[data_path]

                positions_parts.append(np.column_stack([geo['X'][:], geo['Y'][:], geo['Z'][:]]))
                area_parts.append(geo['Area'][:])
                force_parts.append(np.stack([
                    data['Surface_X-Force'][:],
                    data['Surface_Y-Force'][:],
                    data['Surface_Z-Force'][:],
                ], axis=-1))  # (n_frames, n_points_label, 3)

            self.positions = np.concatenate(positions_parts, axis=0)
            self.area = np.concatenate(area_parts, axis=0)
            self.force_per_area = np.concatenate(force_parts, axis=1)  # (n_frames, n_points, 3)

    def _span_chord(self):

        '''Raw Cartesian (span, chord) position, centered - see FrictionLines._span_chord().'''

        span = self.positions[:, self.span_axis]
        chord = self.positions[:, self.chord_axis]
        span = span - (span.min() + span.max()) / 2
        chord = chord - (chord.min() + chord.max()) / 2

        return span, chord

    def _basis(self, flip_axial: bool = False, flip_tangential: bool = False):

        '''
        Per-surfel (radial_unit, axial_unit, tangential_unit, radius) -
        see class docstring's "Physical basis" for what each means and
        the two sign ambiguities flip_axial/flip_tangential correct.
        '''

        rel = self.positions - self.axis_origin
        along = rel @ self.axis_direction
        radial_vec = rel - along[:, None] * self.axis_direction
        radius = np.linalg.norm(radial_vec, axis=1)
        radial_unit = radial_vec / radius[:, None]

        axial_direction = -self.axis_direction if flip_axial else self.axis_direction
        axial_unit = np.broadcast_to(axial_direction, radial_unit.shape)

        tangential_unit = np.cross(axial_unit, radial_unit)
        if flip_tangential:
            tangential_unit = -tangential_unit

        return radial_unit, axial_unit, tangential_unit, radius

    def _selected_forces(self, span_min: float = None, span_max: float = None, min_count: int = 1,
                          flip_axial: bool = False, flip_tangential: bool = False):

        '''
        Shared by compute() and total_loads(): span/chord-crop the blade,
        project every selected surfel's force onto the physical
        (radial, axial, tangential) basis (see _basis()). Both callers
        then either bin this (compute()) or sum it directly (total_loads()).

        Returns
        -------
        span_m, chord_m, radius_m : np.ndarray, shape (n_sel,)
        F_axial, F_radial, F_tangential : np.ndarray, shape (n_frames, n_sel)
        '''

        span, chord = self._span_chord()
        radial_unit, axial_unit, tangential_unit, radius = self._basis(
            flip_axial=flip_axial, flip_tangential=flip_tangential)

        mask = np.ones(len(span), dtype=bool)
        if span_min is not None:
            mask &= span >= span_min
        if span_max is not None:
            mask &= span <= span_max

        if mask.sum() < min_count:
            raise ValueError(
                f"Only {int(mask.sum())} points after span_min/span_max cropping - "
                "check span_min/span_max."
            )

        span_m, chord_m, radius_m = span[mask], chord[mask], radius[mask]
        force_m = self.force_per_area[:, mask, :] * self.area[mask][None, :, None]  # (n_frames, n_sel, 3)

        F_axial = np.einsum('fpc,pc->fp', force_m, axial_unit[mask])
        F_radial = np.einsum('fpc,pc->fp', force_m, radial_unit[mask])
        F_tangential = np.einsum('fpc,pc->fp', force_m, tangential_unit[mask])

        return span_m, chord_m, radius_m, F_axial, F_radial, F_tangential

    @staticmethod
    def _totals_from_forces(F_axial, F_radial, F_tangential, radius_m):

        '''
        Net thrust/radial_force/tangential_force/torque from already-
        projected per-surfel forces - shared by total_loads() (a fresh,
        standalone selection) and compute() (embedded in its own result,
        computed from the EXACT SAME selection/surfels used for the
        strip binning, so the two can never disagree - see compute()'s
        'totals' key).
        '''

        return {
            'thrust': F_axial.sum(axis=1),
            'radial_force': F_radial.sum(axis=1),
            'tangential_force': F_tangential.sum(axis=1),
            'torque': (F_tangential * radius_m[None, :]).sum(axis=1),
        }

    def total_loads(self, span_min: float = None, span_max: float = None,
                     flip_axial: bool = False, flip_tangential: bool = False):

        '''
        Net integrated force (thrust, radial, tangential) and torque,
        summed directly over every selected surfel - independent of
        strip binning entirely (no n_span_bins/n_chord_bins/bin-edge
        choice involved). A standalone way to get the total without
        running compute() first - if you already called compute() and
        want the total for THAT SAME range, use its own 'totals' key
        instead (see compute()'s docstring) rather than calling this
        separately, so there's no risk of passing a different span_min/
        span_max here by mistake and getting numbers that don't match a
        plot built from compute()'s result.

        Validated against this project's own case (one blade,
        span_min=0.02): thrust = 1.319 N and torque = 0.0219 N.m,
        averaged over frames - both consistent with half of the user's
        independently known totals for the full 2-bladed rotor (2.6 N
        thrust, ~0.044 N.m torque).

        Two force totals are reported beyond thrust/torque, for
        completeness/QA:
            'radial_force' : Sum(F_radial) - net force pointing away
                from the rotation axis. Aerodynamically this is usually
                small - most of a real blade's outward pull is
                CENTRIFUGAL, a structural/mass effect this purely
                aerodynamic tool has no way to see. If it's NOT small
                relative to thrust, that's worth a second look (a real
                3D effect, or a bug).
            'tangential_force' : Sum(F_tangential) - the net force in
                the plane of rotation. NOT the same as torque (torque is
                this SAME per-surfel quantity weighted by its own radius
                before summing - a moment, not a raw force sum) - this
                is the net in-plane aerodynamic force, related to (but
                distinct from) the classic rotorcraft in-plane "H-force".

        Parameters
        ----------
        (same span_min/span_max/flip_axial/flip_tangential as compute())

        Returns
        -------
        dict
            'thrust', 'radial_force', 'tangential_force' : np.ndarray,
                shape (n_frames,) - net force [N] in each direction,
                every frame.
            'torque' : np.ndarray, shape (n_frames,) - net moment about
                the rotation axis [N.m], every frame.
        '''

        _, _, radius_m, F_axial, F_radial, F_tangential = self._selected_forces(
            span_min=span_min, span_max=span_max, flip_axial=flip_axial, flip_tangential=flip_tangential)

        return self._totals_from_forces(F_axial, F_radial, F_tangential, radius_m)

    def compute(self, span_min: float = None, span_max: float = None, n_span_bins: int = 20,
                n_chord_bins: int = None, chord_percentile: float = 0.5, min_count: int = 5,
                flip_axial: bool = False, flip_tangential: bool = False):

        '''
        Per-strip (optionally per-strip-per-chord-bin) axial/radial/
        tangential force, every frame in the file.

        Restricted to ONE blade section via span_min/span_max, same
        reason as everywhere else in this project (FrictionLines/
        SurfaceVariable): a multi-blade file's raw span axis mixes
        surfels from every blade at the same radius otherwise, which
        would corrupt the strip sums (a strip's force would silently
        include every blade's contribution, not just one). No default -
        depends on this case's own span layout, check per case.

        Parameters
        ----------
        n_span_bins : int
            Number of radial strips the (cropped) span range is
            partitioned into.
        n_chord_bins : int, optional
            If set, each radial strip is further subdivided into this
            many chordwise sub-strips (see class docstring's "Chordwise
            (non-compactness) subdivision"). None (default): one strip
            per radial band, matching the original PowerVIZ-based
            per-strip export.
        chord_percentile : float
            Only used if n_chord_bins is set - each strip's chordwise
            extent is the [chord_percentile, 100-chord_percentile]
            percentile of chord within it, not literal min/max - same
            outlier-guarding purpose as elsewhere in this project.
        min_count : int
            Minimum surfels a (span [, chord]) bin needs to be trusted -
            bins with fewer are left as NaN (radius/chord) or 0 (force -
            genuinely no surfels there contributes exactly zero force,
            which is the physically correct value, unlike a NaN-worthy
            "unreliable average").
        flip_axial, flip_tangential : bool
            See class docstring - swap either sign if it comes out
            backwards from what you expect physically for this case.

        Returns
        -------
        dict
            'radius' : np.ndarray, shape (n_span_bins,) - mean physical
                radius [m] of each strip (NaN where empty).
            'chord' : np.ndarray, shape (n_span_bins, n_chord_bins), or
                None if n_chord_bins wasn't set - mean chord position
                [m, centered/raw Cartesian - see _span_chord()] of each
                chordwise sub-bin (NaN where empty).
            'axial', 'radial', 'tangential' : np.ndarray, shape
                (n_frames, n_span_bins) or (n_frames, n_span_bins,
                n_chord_bins) if n_chord_bins was set - net force [N] in
                each direction, summed over that bin's surfels, every
                frame.
            'totals' : dict - thrust/radial_force/tangential_force/
                torque (see total_loads()), computed from the EXACT SAME
                selection (span_min/span_max, flip_axial/flip_tangential)
                as the strip breakdown above - guaranteed consistent
                with it (e.g. for plot_bar_forces(show_totals=True)'s
                annotation), unlike calling total_loads() separately
                with its own span_min/span_max, which could silently
                drift out of sync if given different values by mistake.
        '''

        span_m, chord_m, radius_m, F_axial, F_radial, F_tangential = self._selected_forces(
            span_min=span_min, span_max=span_max, min_count=min_count,
            flip_axial=flip_axial, flip_tangential=flip_tangential)

        totals = self._totals_from_forces(F_axial, F_radial, F_tangential, radius_m)

        span_edges = np.linspace(span_m.min(), span_m.max(), n_span_bins + 1)
        radius_out = np.full(n_span_bins, np.nan)

        if n_chord_bins is None:
            axial_out = np.zeros((self.n_frames, n_span_bins))
            radial_out = np.zeros((self.n_frames, n_span_bins))
            tangential_out = np.zeros((self.n_frames, n_span_bins))
            chord_out = None
        else:
            axial_out = np.zeros((self.n_frames, n_span_bins, n_chord_bins))
            radial_out = np.zeros((self.n_frames, n_span_bins, n_chord_bins))
            tangential_out = np.zeros((self.n_frames, n_span_bins, n_chord_bins))
            chord_out = np.full((n_span_bins, n_chord_bins), np.nan)

        for i in range(n_span_bins):

            in_bin = (span_m >= span_edges[i]) & (
                span_m < span_edges[i + 1] if i < n_span_bins - 1 else span_m <= span_edges[i + 1]
            )
            n_sel = int(in_bin.sum())
            if n_sel < min_count:
                continue

            radius_out[i] = radius_m[in_bin].mean()

            if n_chord_bins is None:
                axial_out[:, i] = F_axial[:, in_bin].sum(axis=1)
                radial_out[:, i] = F_radial[:, in_bin].sum(axis=1)
                tangential_out[:, i] = F_tangential[:, in_bin].sum(axis=1)
                continue

            c = chord_m[in_bin]
            c_min, c_max = np.percentile(c, [chord_percentile, 100 - chord_percentile])
            c_edges = np.linspace(c_min, c_max, n_chord_bins + 1)
            c_idx = np.clip(np.digitize(c, c_edges) - 1, 0, n_chord_bins - 1)

            one_hot = np.zeros((n_sel, n_chord_bins))
            one_hot[np.arange(n_sel), c_idx] = 1
            counts = one_hot.sum(axis=0)
            has_data = counts >= min_count

            sums_c = c @ one_hot
            chord_out[i, has_data] = sums_c[has_data] / counts[has_data]

            axial_out[:, i, :] = F_axial[:, in_bin] @ one_hot
            radial_out[:, i, :] = F_radial[:, in_bin] @ one_hot
            tangential_out[:, i, :] = F_tangential[:, in_bin] @ one_hot
            axial_out[:, i, ~has_data] = 0.0
            radial_out[:, i, ~has_data] = 0.0
            tangential_out[:, i, ~has_data] = 0.0

        return {
            'radius': radius_out, 'chord': chord_out,
            'axial': axial_out, 'radial': radial_out, 'tangential': tangential_out,
            'totals': totals,
        }

    def save(self, result: dict, filepath: str, dt: float = None):

        '''
        Write compute()'s result to an independent HDF5 file - Hanson's
        method's raw input (per-strip, optionally per-strip-per-chord-
        bin, time-resolved axial/radial/tangential sectional loading),
        usable on its own without needing this class or the original
        .snc-derived file again.

        Parameters
        ----------
        dt : float, optional
            Physical timestep [s] between consecutive frames - if given,
            written as Time/Time (frame_index * dt), same convention as
            converters.forces_strip.ForcesCSVConverter's dt parameter.
            SNCReader carries no real timestep info on its own (see
            README.md) - omit if unknown; the frame axis is still there
            (Data/axial etc.'s first dimension), just without a physical
            time value attached.
        '''

        with h5py.File(filepath, 'w') as f:

            geo = f.create_group('Geometry')
            geo.create_dataset('radius', data=result['radius'])
            if result['chord'] is not None:
                geo.create_dataset('chord', data=result['chord'])

            if dt is not None:
                f.create_dataset('Time/Time', data=np.arange(self.n_frames) * dt)

            data = f.create_group('Data')
            data.create_dataset('axial', data=result['axial'])
            data.create_dataset('radial', data=result['radial'])
            data.create_dataset('tangential', data=result['tangential'])

    def plot_bar_forces(self, result: dict, frame: int = None, ax=None, bar_width: float = None,
                         colors=('tab:blue', 'tab:orange', 'tab:green'), show_totals: bool = False,
                         normalize_radius: bool = True, rho: float = None, n_rot: float = None,
                         diameter: float = None, savepath: str = None, dpi: int = 150):

        '''
        Bar chart of per-strip force (left axis) plus its running
        cumulative sum across strips, root to tip (right axis, dashed
        lines) - matches this project's earlier PowerVIZ-based reference
        plot (images/forces/bar_forces/Force-Graph-1-bar_forces.png),
        for direct visual comparison against that established style.
        Only meaningful for n_chord_bins=None results (one bar per
        radial strip) - a chord-subdivided result has more than one
        value per strip and doesn't reduce to a single bar chart this
        way.

        frame : int or None
            A specific frame index for an instantaneous bar chart, or
            None (default) to average over every frame in the file
            first - same convention as FrictionLines/SurfaceVariable.
        show_totals : bool
            If True, annotate the figure with a text box, same style as
            FrictionLines' Poincare-index annotation, in the lower-left
            (near the root, where the bars themselves stay small - see
            the class's own validated example - so the box doesn't sit
            on top of data). Reads compute()'s OWN embedded
            result['totals'], not a fresh total_loads() call -
            guaranteed to match the exact span_min/span_max this
            particular result was built with (see compute()'s docstring
            for why that matters).
                Without rho/n_rot/diameter: shows the dimensional
                totals - thrust [N], torque [N.m], radial_force [N],
                tangential_force [N].
                WITH rho/n_rot/diameter (see below): shows ONLY the
                non-dimensional coefficients (C_T,axial/C_T,radial/
                C_T,tangential/C_Q) - the dimensional N/N.m values are
                deliberately NOT shown alongside them in this mode, for
                cases where the raw loads themselves shouldn't be
                exposed (e.g. confidentiality) but their non-dimensional
                form is fine to share.
        normalize_radius : bool
            If True (default), the x-axis is r/R (needs r_tip - see
            __init__). If False, physical radius [m].
        rho, n_rot, diameter : float, optional
            Density [kg/m^3], rotation rate [rev/s - NOT rad/s], and
            rotor diameter [m]. If all three are given, the y-axis
            (bars AND the cumulative curve) is non-dimensionalized as a
            standard propeller-convention force coefficient, the SAME
            equation applied to all three force components:

                C_T,axial      = thrust           / (rho * n_rot^2 * diameter^4)
                C_T,radial     = radial_force     / (rho * n_rot^2 * diameter^4)
                C_T,tangential = tangential_force / (rho * n_rot^2 * diameter^4)

            Deliberately NOT a separate "torque coefficient" for the
            tangential bar: what's plotted there is still a per-strip/
            cumulative FORCE, not torque (a moment - see
            total_loads()'s docstring on why 'torque' needs a radius-
            weighted sum, not a plain force sum) - dividing it by the
            same force-normalization as thrust keeps it dimensionally
            honest as "just another force coefficient", not torque's own
            coefficient. The PROPER torque coefficient, computed from
            total_loads()'s radius-weighted torque (not the tangential
            bar), uses one extra factor of diameter, since torque itself
            carries one extra length dimension (force x lever arm) that
            a plain force doesn't:

                C_Q = torque / (rho * n_rot^2 * diameter^5)

            Both only appear in the show_totals box (see above), never
            as their own bar/line on the chart.

        Returns
        -------
        (fig, (ax, ax2))
        '''

        if result['chord'] is not None:
            raise ValueError(
                "plot_bar_forces() only supports a per-radial-strip result (compute(n_chord_bins=None)) "
                "- this result has chordwise sub-bins, which don't reduce to one bar per strip."
            )

        if normalize_radius and self.r_tip is None:
            raise ValueError("r_tip must be set (in __init__) to plot the x-axis as r/R (normalize_radius=True).")

        coefficients = rho is not None and n_rot is not None and diameter is not None
        norm_force = rho * n_rot ** 2 * diameter ** 4 if coefficients else 1.0
        norm_torque = rho * n_rot ** 2 * diameter ** 5 if coefficients else 1.0

        radius = result['radius']
        valid = ~np.isnan(radius)
        order = np.argsort(radius[valid])

        def reduce(arr):
            per_frame = arr if frame is None else arr[frame:frame + 1]
            return np.nanmean(per_frame, axis=0)[valid][order]

        radius_sorted = radius[valid][order] / self.r_tip if normalize_radius else radius[valid][order]
        axial = reduce(result['axial']) / norm_force
        radial = reduce(result['radial']) / norm_force
        tangential = reduce(result['tangential']) / norm_force

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if bar_width is None:
            bar_width = 0.6 * np.median(np.diff(radius_sorted))

        ax.bar(radius_sorted, radial, width=bar_width, color=colors[0], label='Radial Force', zorder=2)
        ax.bar(radius_sorted, axial, width=bar_width, color=colors[1], label='Axial Force', zorder=1)
        ax.bar(radius_sorted, tangential, width=bar_width, color=colors[2], label='Tangential Force', zorder=3)

        ax2 = ax.twinx()
        ax2.plot(radius_sorted, np.cumsum(radial), '--', color=colors[0])
        ax2.plot(radius_sorted, np.cumsum(axial), '--', color=colors[1])
        ax2.plot(radius_sorted, np.cumsum(tangential), '--', color=colors[2])

        ax.set_xlabel(r'$r/R$ [-]' if normalize_radius else 'Radius [m]')
        ax.set_ylabel(r'$C_F$ [-]' if coefficients else 'Force [N]')
        ax2.set_ylabel(r'$C_{F,\Sigma}$ [-]' if coefficients else 'Integrated Force [N]')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left')

        if show_totals:
            totals = result['totals']

            def reduce_scalar(arr):
                per_frame = arr if frame is None else arr[frame:frame + 1]
                return np.mean(per_frame)

            if coefficients:
                # Coefficients only - no dimensional N/N.m values, by
                # design: this mode exists specifically for cases where
                # the raw dimensional loads shouldn't be shown/shared
                # (e.g. confidentiality), only their non-dimensional form.
                note = (
                    f"$C_{{T,axial}}$ = {reduce_scalar(totals['thrust']) / norm_force:.4g}\n"
                    f"$C_{{T,radial}}$ = {reduce_scalar(totals['radial_force']) / norm_force:.4g}\n"
                    f"$C_{{T,tangential}}$ = {reduce_scalar(totals['tangential_force']) / norm_force:.4g}\n"
                    f"$C_Q$ = {reduce_scalar(totals['torque']) / norm_torque:.4g}"
                )
            else:
                note = (
                    f"Thrust = {reduce_scalar(totals['thrust']):.4g} N\n"
                    f"Torque = {reduce_scalar(totals['torque']):.4g} N.m\n"
                    f"Radial force = {reduce_scalar(totals['radial_force']):.4g} N\n"
                    f"Tangential force = {reduce_scalar(totals['tangential_force']):.4g} N"
                )
            # lower-left: near the root, where the bars themselves stay
            # small (see plot), so the box doesn't sit on top of data -
            # and clear of the legend, which occupies the upper-left.
            ax.text(0.02, 0.35, note, transform=ax.transAxes, fontsize=11, va='center', ha='left',
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.85))

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, (ax, ax2)

    def _time_axis(self, dt: float):

        '''Physical time [s] for every frame - see save()'s dt parameter (SNCReader carries no real timestep of its own).'''

        return np.arange(self.n_frames) * dt

    _COMPONENT_LABELS = {
        'axial': r'$F_{axial}$ [N]',
        'radial': r'$F_{radial}$ [N]',
        'tangential': r'$F_{tangential}$ [N]',
    }

    def plot_time_trace(self, result: dict, dt: float, component: str = 'axial', strips=None,
                         ax=None, cmap: str = 'cividis', savepath: str = None, dpi: int = 150):

        '''
        Raw per-strip force vs time - one line per strip (color-coded),
        the time-domain view of compute()'s per-frame result - matches
        this project's StripForces_vs_time.png reference style.

        Only meaningful for an "inst" (multi-frame/transient) file, NOT
        an "average" one (a single, already time-averaged frame has
        nothing to trace) - raises if compute()'s file only had 1 frame.

        Parameters
        ----------
        result : dict
            compute()'s return value - per-radial-strip only
            (n_chord_bins=None), same restriction as plot_bar_forces().
        dt : float
            Physical timestep [s] between frames.
        component : 'axial', 'radial', or 'tangential'
        strips : array-like of int, optional
            Which strip INDICES (0-based, into result['radius']) to
            plot - None (default) plots every valid strip. A real case
            can have far more strips than are legible on one plot at
            once - use this to pick a representative subset, matching
            the reference's 8-strip example.

        Returns
        -------
        (fig, ax)
        '''

        if result['chord'] is not None:
            raise ValueError(
                "plot_time_trace() only supports a per-radial-strip result (compute(n_chord_bins=None))."
            )
        if self.n_frames < 2:
            raise ValueError(
                f"Need at least 2 frames for a time trace - this file has only {self.n_frames} "
                "(looks like an already-averaged 'average' case, not an 'inst' one)."
            )

        t = self._time_axis(dt)
        radius = result['radius']
        valid = np.flatnonzero(~np.isnan(radius))
        sel = valid if strips is None else np.asarray(strips)

        vals = result[component]  # (n_frames, n_span_bins)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(sel)))
        for color, i in zip(colors, sel):
            ax.plot(t, vals[:, i], color=color, label=f'Strip {i + 1}')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel(self._COMPONENT_LABELS[component])
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax

    def phase_lock(self, result: dict, dt: float, n_azimuth_bins: int = 72):

        '''
        Phase-locked (revolution-folded) average of compute()'s
        per-frame strip forces - bins every frame by its rotor azimuth
        angle (needs rpm, set in __init__, and dt, same as elsewhere -
        SNCReader carries no real time info of its own) into
        n_azimuth_bins bins spanning one full revolution (0-360 deg),
        and averages every frame that lands in each bin, however many
        revolutions that spans in total - the standard "phase-locked
        averaging" used to see a periodic load's once-per-rev pattern
        cleanly, averaged over turbulence/frame-to-frame noise. Matches
        this project's StripForces_vs_angle.png / Strip_PhasedLocked.pdf
        reference style (see plot_vs_angle()).

        Needs at least roughly one full revolution of frames to be
        meaningful - fewer leaves most azimuth bins empty (NaN); more
        (several revolutions) is what actually makes the AVERAGING part
        do anything (a single revolution's worth of frames just
        re-bins each frame into its own bin with no folding).

        Azimuth is computed as `(t * rpm * 6) mod 360` (`rpm * 6` =
        `rpm * 360/60`, degrees per second) - this assumes CONSTANT rpm
        across every frame in the file (true for the steady-RPM cases
        this project works with so far; a genuinely variable-rpm run
        would need a real per-frame azimuth log instead, which isn't
        available from this file either way - see README.md's notes on
        SNCReader carrying no per-frame time metadata).

        Parameters
        ----------
        result : dict
            compute()'s return value (any n_chord_bins).
        dt : float
            Physical timestep [s] between frames.
        n_azimuth_bins : int
            Number of azimuth bins spanning 0-360 degrees.

        Returns
        -------
        dict
            'azimuth_deg' : np.ndarray, shape (n_azimuth_bins,) - bin
                center azimuth angles [deg].
            'radius', 'chord' : passed through from result, unchanged.
            'axial', 'radial', 'tangential' : np.ndarray, shape
                (n_azimuth_bins, n_span_bins[, n_chord_bins]) -
                phase-locked mean force [N], NaN where no frame fell in
                that azimuth bin for that strip.
        '''

        if self.rpm is None:
            raise ValueError("rpm must be set (in __init__) to compute azimuth angle for phase-locking.")
        if self.n_frames < 2:
            raise ValueError(f"Need at least 2 frames to phase-lock - this file has only {self.n_frames}.")

        t = self._time_axis(dt)
        azimuth = (t * self.rpm * 6.0) % 360.0

        edges = np.linspace(0, 360, n_azimuth_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_idx = np.clip(np.digitize(azimuth, edges) - 1, 0, n_azimuth_bins - 1)
        counts = np.bincount(bin_idx, minlength=n_azimuth_bins).astype(float)
        has_data = counts > 0

        out = {'azimuth_deg': centers, 'radius': result['radius'], 'chord': result['chord']}

        for key in ('axial', 'radial', 'tangential'):
            arr = result[key]
            shape_rest = arr.shape[1:]
            flat = arr.reshape(self.n_frames, -1)

            sums = np.zeros((n_azimuth_bins, flat.shape[1]))
            np.add.at(sums, bin_idx, flat)

            mean = np.full_like(sums, np.nan)
            mean[has_data] = sums[has_data] / counts[has_data, None]
            out[key] = mean.reshape((n_azimuth_bins,) + shape_rest)

        return out

    def plot_vs_angle(self, phase_locked: dict, component: str = 'axial', strips=None, polar: bool = True,
                       ax=None, cmap: str = 'cividis', savepath: str = None, dpi: int = 150):

        '''
        Phase-locked force vs rotor azimuth - matches this project's
        StripForces_vs_angle.png reference style (polar, one curve per
        strip, colors matching plot_time_trace()'s if the same
        cmap/strip selection is reused). Takes phase_lock()'s result,
        NOT compute()'s own raw per-frame series directly - a raw time
        series doesn't have one value per azimuth angle the way
        phase_lock()'s folded/averaged result does.

        polar : bool
            True (default): polar plot (radial axis = force, angular
            axis = azimuth, 0deg at 3 o'clock/east, matching the
            reference). False: a plain Cartesian plot (force vs azimuth
            in degrees, 0-360).

        Returns
        -------
        (fig, ax)
        '''

        if phase_locked['chord'] is not None:
            raise ValueError("plot_vs_angle() only supports a per-radial-strip phase_lock() result.")

        radius = phase_locked['radius']
        valid = np.flatnonzero(~np.isnan(radius))
        sel = valid if strips is None else np.asarray(strips)

        azimuth_deg = phase_locked['azimuth_deg']
        vals = phase_locked[component]  # (n_azimuth_bins, n_span_bins)

        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='polar' if polar else None)
        else:
            fig = ax.figure

        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(sel)))
        theta = np.deg2rad(azimuth_deg)
        if polar:
            # close the loop - wrap the last point back to the first so
            # the curve doesn't leave a visible gap at 0/360 deg.
            theta = np.append(theta, theta[0])

        for color, i in zip(colors, sel):
            y = vals[:, i]
            if polar:
                y = np.append(y, y[0])
                ax.plot(theta, y, color=color, label=f'Strip {i + 1}')
            else:
                ax.plot(azimuth_deg, y, color=color, label=f'Strip {i + 1}')

        ylabel = self._COMPONENT_LABELS[component]

        if polar:
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.set_ylabel(ylabel, labelpad=30)
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05))
        else:
            ax.set_xlabel(r'$\phi$ [$^\circ$]')
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax

    def harmonics(self, result: dict, dt: float, component: str = 'axial', n_harmonics: int = 20,
                  detrend: bool = True, return_phase: bool = False):

        '''
        Fourier decomposition of each strip's force time history into
        harmonics of the rotor's OWN rotation frequency (1P, 2P, 3P,
        ...) - the |F_n(r)| input Hanson's tonal noise method needs
        (matches this project's
        Strips1-4_histogram_compare_logscale.png reference style -
        "BPF and harmonics" there is this same per-revolution harmonic
        series, computed per-blade, in the ROTATING frame; combining
        these per-blade sectional harmonics into an observer's actual
        blade-passage-frequency tones - at n_blades times this rotation
        frequency - is Hanson's model's own downstream step, not
        computed here).

        Uses a plain FFT (numpy.fft.rfft) over the WHOLE time series at
        once - not a windowed/averaged method like Welch's (see
        SurfaceVariable.periodogram()) - because this wants the actual
        per-revolution harmonic content of a (quasi-)periodic unsteady
        load, not a smoothed power spectral density. This means the file
        should span close to an integer number of full revolutions for
        the harmonic bins to land exactly on integer multiples of the
        rotation frequency - a partial revolution smears each peak
        across neighboring FFT bins (classic spectral leakage); this
        method does NOT correct for that, it just reports the FFT bin
        nearest each harmonic's exact target frequency.

        Parameters
        ----------
        result : dict
            compute()'s return value - per-radial-strip only
            (n_chord_bins=None), same restriction as plot_bar_forces().
        dt : float
            Physical timestep [s] between frames.
        component : 'axial', 'radial', or 'tangential'
        n_harmonics : int
            Number of harmonics (1P, 2P, ..., n_harmonics P) to return -
            capped automatically at what the frame count/dt can actually
            resolve (Nyquist: rotation frequency * n_harmonics <= 1/(2 dt)).
        detrend : bool
            Remove each strip's own mean before the FFT (default True) -
            a nonzero mean shows up at 0P (DC), not a real harmonic;
            doesn't change 1P and above, but keeps a large DC offset
            from dominating the FFT's overall scaling.
        return_phase : bool
            If True, also return each harmonic's PHASE (radians, from
            `numpy.angle`, in `(-pi, pi]`) alongside its magnitude - see
            this method's own "What the phase is for" note below and
            `peak_azimuth()`/`reconstruct_from_harmonics()`, which both
            need it. Off by default - Hanson's radiated-noise step needs
            phase eventually (a magnitude-only harmonic can't be summed
            back into a physically meaningful signal, and interference
            between radial stations depends on their RELATIVE phase),
            but a lot of exploratory work (this project's own
            Strips1-4_histogram_compare_logscale.png reference among
            them) only ever shows/uses magnitude - kept optional so the
            common case doesn't carry the extra array around unused.

        What the phase is for (beyond just being a required Hanson
        input): each harmonic's phase pins down WHERE in the revolution
        (what azimuth) that harmonic's own peak sits - useful on its
        own, independent of the final noise calculation, for relating a
        loading harmonic back to a physical cause (e.g. this project's
        own strut-interaction case: the 1P phase should point at the
        strut's azimuth if that's really what's driving it). See
        peak_azimuth() and reconstruct_from_harmonics().

        Returns
        -------
        dict
            'radius', 'chord' : passed through from result, unchanged.
            'harmonic' : np.ndarray, shape (n_harmonics,) - 1, 2, 3, ...
                (1P, 2P, ...).
            'magnitude' : np.ndarray, shape (n_harmonics, n_span_bins) -
                |F_n(r)| [N] at each harmonic, each strip.
            'phase' : np.ndarray, shape (n_harmonics, n_span_bins) -
                only present if return_phase=True - phase [rad] at each
                harmonic, each strip.
        '''

        if self.rpm is None:
            raise ValueError("rpm must be set (in __init__) to compute harmonics of the rotation frequency.")
        if self.n_frames < 4:
            raise ValueError(f"Need several frames for a meaningful FFT - this file has only {self.n_frames}.")
        if result['chord'] is not None:
            raise ValueError(
                "harmonics() only supports a per-radial-strip result (compute(n_chord_bins=None))."
            )

        arr = result[component]  # (n_frames, n_span_bins)

        flat = arr - arr.mean(axis=0, keepdims=True) if detrend else arr

        freq = np.fft.rfftfreq(self.n_frames, d=dt)
        fft_vals = np.fft.rfft(flat, axis=0) / self.n_frames
        magnitude = 2 * np.abs(fft_vals)  # single-sided amplitude
        phase = np.angle(fft_vals)

        rotation_freq = self.rpm / 60.0
        max_harmonic = int(freq.max() / rotation_freq)
        n_harmonics = min(n_harmonics, max_harmonic)
        if n_harmonics < 1:
            raise ValueError(
                f"Sampling rate too low (dt={dt}, rpm={self.rpm}) to resolve even the 1st harmonic of "
                "the rotation frequency - check dt/rpm, or this file doesn't have enough frames."
            )

        harmonic_numbers = np.arange(1, n_harmonics + 1)
        target_freqs = harmonic_numbers * rotation_freq
        bin_idx = np.array([np.argmin(np.abs(freq - tf)) for tf in target_freqs])

        out = {
            'radius': result['radius'], 'chord': result['chord'],
            'harmonic': harmonic_numbers,
            'magnitude': magnitude[bin_idx],
        }
        if return_phase:
            out['phase'] = phase[bin_idx]

        return out

    def peak_azimuth(self, harmonics_result: dict):

        '''
        Azimuth [deg, 0-360) where each harmonic's own contribution
        peaks - `phi_peak = -phase/n mod (360/n)` (a harmonic `n` repeats
        `n` times per revolution, so it has `n` equally-spaced peaks;
        this reports the first one). Useful for tying a loading harmonic
        back to a physical cause - e.g. checking whether the 1P peak
        azimuth lines up with a known disturbance's position (this
        project's own strut-interaction reference case, for instance).

        Needs harmonics_result['phase'] - call harmonics(..., return_phase=True) first.

        Returns
        -------
        np.ndarray, shape (n_harmonics, n_span_bins) - peak azimuth [deg].
        '''

        if 'phase' not in harmonics_result:
            raise ValueError("harmonics_result has no 'phase' - call harmonics(..., return_phase=True) first.")

        harmonic = harmonics_result['harmonic'][:, None]
        phase = harmonics_result['phase']

        return (-np.rad2deg(phase) / harmonic) % (360.0 / harmonic)

    def reconstruct_from_harmonics(self, harmonics_result: dict, azimuth_deg=None):

        '''
        Rebuild an azimuth-domain curve from harmonics()'s magnitude +
        phase - `sum_n magnitude_n * cos(n*phi - phase_n)` (the DC/mean
        term isn't included, since harmonics() detrends it away by
        default - add the strip's own time-mean back separately if you
        want an absolute-level reconstruction, not just the fluctuating
        part).

        The main use: overlay this against phase_lock()'s own empirical
        folded curve (same azimuth axis) as a validation check - if a
        handful of harmonics already reconstructs the real curve
        closely, that confirms the FFT decomposition actually captured
        the dominant unsteady content (and gives you an honest sense of
        how many harmonics matter for this case) rather than trusting
        the harmonic magnitudes/phases blind.

        Needs harmonics_result['phase'] - call harmonics(..., return_phase=True) first.

        Parameters
        ----------
        azimuth_deg : array-like, optional
            Azimuth values [deg] to evaluate the reconstruction at -
            defaults to a dense 0-360 sweep (361 points) if not given;
            pass phase_lock()'s own 'azimuth_deg' to compare bin-for-bin.

        Returns
        -------
        azimuth_deg, reconstructed : np.ndarray, np.ndarray
            reconstructed shape (len(azimuth_deg), n_span_bins).
        '''

        if 'phase' not in harmonics_result:
            raise ValueError("harmonics_result has no 'phase' - call harmonics(..., return_phase=True) first.")

        if azimuth_deg is None:
            azimuth_deg = np.linspace(0, 360, 361)
        azimuth_deg = np.asarray(azimuth_deg)

        phi = np.deg2rad(azimuth_deg)[:, None, None]              # (n_az, 1, 1)
        n = harmonics_result['harmonic'][None, :, None]            # (1, n_harmonics, 1)
        mag = harmonics_result['magnitude'][None, :, :]            # (1, n_harmonics, n_span_bins)
        phase = harmonics_result['phase'][None, :, :]               # (1, n_harmonics, n_span_bins)

        reconstructed = np.sum(mag * np.cos(n * phi - phase), axis=1)  # (n_az, n_span_bins)

        return azimuth_deg, reconstructed

    def save_harmonics(self, harmonics_result: dict, filepath: str):

        '''
        Write harmonics()'s result to an independent HDF5 file - ready
        to hand off as Hanson's method's actual loading-harmonic input,
        without needing this class or the original .snc-derived file
        again. Includes 'phase' only if harmonics_result has it (i.e.
        harmonics(..., return_phase=True) was used).
        '''

        with h5py.File(filepath, 'w') as f:

            geo = f.create_group('Geometry')
            geo.create_dataset('radius', data=harmonics_result['radius'])
            if harmonics_result['chord'] is not None:
                geo.create_dataset('chord', data=harmonics_result['chord'])

            data = f.create_group('Data')
            data.create_dataset('harmonic', data=harmonics_result['harmonic'])
            data.create_dataset('magnitude', data=harmonics_result['magnitude'])
            if 'phase' in harmonics_result:
                data.create_dataset('phase', data=harmonics_result['phase'])

    def plot_harmonics(self, harmonics_result: dict, strips=None, show_phase: bool = False, ax=None,
                        cmap: str = 'cividis', savepath: str = None, dpi: int = 150):

        '''
        Bar chart of harmonics()'s |F_n(r)| vs harmonic number, log
        y-axis, grouped bars (one color per strip) - matches this
        project's Strips1-4_histogram_compare_logscale.png reference
        style.

        show_phase : bool
            If True, add a second panel below showing each harmonic's
            phase [deg] the same way (needs
            harmonics_result['phase'] - call harmonics(...,
            return_phase=True) first). Off by default: most of the time
            (including this project's own reference plot) only the
            magnitude is shown - phase matters once you're actually
            about to hand this off downstream (see harmonics()'s "What
            the phase is for" note), not for a first look at the
            spectrum.

        Returns
        -------
        (fig, ax) if show_phase is False, else (fig, (ax, ax_phase))
        '''

        if harmonics_result['chord'] is not None:
            raise ValueError("plot_harmonics() only supports a per-radial-strip harmonics() result.")
        if show_phase and 'phase' not in harmonics_result:
            raise ValueError(
                "show_phase=True needs harmonics_result['phase'] - call harmonics(..., return_phase=True) first."
            )

        radius = harmonics_result['radius']
        valid = np.flatnonzero(~np.isnan(radius))
        sel = valid if strips is None else np.asarray(strips)

        harmonic = harmonics_result['harmonic']
        mag = harmonics_result['magnitude']  # (n_harmonics, n_span_bins)

        if ax is None:
            if show_phase:
                fig, (ax, ax_phase) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax_phase = None
        else:
            fig = ax.figure
            ax_phase = None

        n_bars = len(sel)
        width = 0.8 / n_bars
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_bars))

        for k, (color, i) in enumerate(zip(colors, sel)):
            offset = (k - (n_bars - 1) / 2) * width
            ax.bar(harmonic + offset, mag[:, i], width=width, color=color, label=f'Strip {i + 1}')

        ax.set_yscale('log')
        ax.set_ylabel(r'$|F_n|$ [N]')
        ax.set_xticks(harmonic)
        ax.grid(True, which='both', axis='y', alpha=0.3)
        ax.legend()

        if show_phase and ax_phase is not None:
            phase_deg = np.rad2deg(harmonics_result['phase'])
            for k, (color, i) in enumerate(zip(colors, sel)):
                offset = (k - (n_bars - 1) / 2) * width
                ax_phase.bar(harmonic + offset, phase_deg[:, i], width=width, color=color)
            ax_phase.set_xlabel('Harmonic ($n$P)')
            ax_phase.set_ylabel(r'$\angle F_n$ [$^\circ$]')
            ax_phase.set_ylim(-180, 180)
            ax_phase.grid(True, alpha=0.3)
        else:
            ax.set_xlabel('Harmonic ($n$P)')

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return (fig, (ax, ax_phase)) if show_phase else (fig, ax)
