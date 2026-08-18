import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18
})


class SurfaceVariable:

    '''
    Per-radius extraction/plotting for ANY named surface variable stored
    under Data/Upper,Lower/<name> - generalizes FrictionLines.cf_at_radii()
    beyond wall shear/Cf to Cp, y+, RMS statistics of anything, or any raw
    variable as-is. Works against either data source, since both share the
    same schema (Geometry/Upper,Lower positions, Data/Upper,Lower/<var>
    with a frame axis, Metadata/lrf_axis_origin,lrf_axis_direction):

        - converters.snc_reader.SNCReader.to_h5(..., surface_split=True)
          (forces branch - Skin_Friction, Surface_X/Y/Z-Force, ...)
        - converters.ensight_to_h5.convert_snc_to_h5(..., surface_split=True)
          (pressure branch - static_pressure, y+ if pf2ens exported it, ...)

    NOT for wall shear/Cf itself - see FrictionLines, which needs normals
    and the tau = F - (F.n)n derivation this class doesn't do.

    Span/chord/radius conventions (raw Cartesian span_axis/chord_axis,
    rotation-axis-based radius) are identical to FrictionLines - see that
    class's docstring for the full reasoning/caveats (in particular: this
    assumes little blade twist, since chordwise position is a raw
    Cartesian axis, not derived from the blade's actual local chord line).
    Duplicated here rather than shared via inheritance, to keep both
    classes simple and self-contained - flag it if you'd rather factor the
    shared geometry logic into a common base.

    Parameters
    ----------
    filename : str
        Path to a to_h5()/convert_snc_to_h5() HDF5 file (surface_split=True).
    r_tip : float, optional
        Physical tip radius [m], for reference only.
    rho_ref, rpm, pref : float, optional
        Reference density [kg/m^3], rotor speed [rev/min], and reference
        (freestream) pressure [Pa] - needed for cp(stat='mean') (all
        three) and cp(stat='rms'/'raw_rms') (rho_ref, rpm only - a
        constant pref doesn't affect a fluctuation's statistics). Not
        needed by variable() itself (raw access).
    span_axis, chord_axis, thickness_axis : int
        Which raw position column (0=X, 1=Y, 2=Z) is spanwise, chordwise,
        thickness-wise - see FrictionLines' docstring.
    '''

    def __init__(self, filename: str, r_tip: float = None, rho_ref: float = None, rpm: float = None,
                 pref: float = None, span_axis: int = 0, chord_axis: int = 2, thickness_axis: int = 1):

        self.filename = filename
        self.r_tip = r_tip
        self.rho_ref = rho_ref
        self.rpm = rpm
        self.pref = pref
        self.span_axis = span_axis
        self.chord_axis = chord_axis
        self.thickness_axis = thickness_axis
        self._load()

    def _load(self):

        '''Load geometry/axis metadata and list available Data/<surface> variable names.'''

        with h5py.File(self.filename, 'r') as f:

            if 'Upper' not in f['Geometry'] or 'Lower' not in f['Geometry']:
                raise ValueError(
                    f"'{self.filename}' has no Geometry/Upper or Geometry/Lower group - "
                    "was it written with surface_split=True?"
                )

            axis_origin = f['Metadata/lrf_axis_origin'][:]
            axis_direction = f['Metadata/lrf_axis_direction'][:]
            self.axis_origin = axis_origin
            self.axis_direction = axis_direction / np.linalg.norm(axis_direction)
            self.n_frames = f['Metadata/frame_index'].shape[0]

            # Real per-frame time [s], only written by ensight_to_h5.py
            # when an nc_stats file was supplied at conversion time -
            # otherwise absent or all-NaN (converters/ensight_to_h5.py,
            # parse_nc_stats()); SNCReader.to_h5() never writes this at
            # all. None here means "no physical time axis available" -
            # see timetrace()/periodogram()'s dt parameter for the
            # fallback.
            if 'Metadata/mid_s' in f:
                mid_s = f['Metadata/mid_s'][:]
                self.frame_times = None if np.all(np.isnan(mid_s)) else mid_s
            else:
                self.frame_times = None

            self.positions = {}
            self.available_variables = {}
            for label in ('Upper', 'Lower'):
                geo = f[f'Geometry/{label}']
                self.positions[label] = np.column_stack([geo['X'][:], geo['Y'][:], geo['Z'][:]])
                self.available_variables[label] = list(f[f'Data/{label}'].keys())

    def _radius(self, surface: str):

        '''Physical radius from the rotor's rotation axis, for every surfel - see FrictionLines._radius().'''

        positions = self.positions[surface]
        rel = positions - self.axis_origin
        along = rel @ self.axis_direction
        radial_vec = rel - along[:, None] * self.axis_direction

        return np.linalg.norm(radial_vec, axis=1)

    def _span_chord(self, surface: str):

        '''Raw Cartesian (span, chord) position, centered - see FrictionLines._span_chord().'''

        positions = self.positions[surface]
        span = positions[:, self.span_axis]
        chord = positions[:, self.chord_axis]
        span = span - (span.min() + span.max()) / 2
        chord = chord - (chord.min() + chord.max()) / 2

        return span, chord

    def _q_ref(self, surface: str):

        '''Local dynamic pressure q_ref = 0.5*rho_ref*(omega*r)^2 - see FrictionLines._q_ref().'''

        omega = self.rpm * 2 * np.pi / 60
        r = self._radius(surface)

        return 0.5 * self.rho_ref * (omega * r) ** 2

    def variable(self, name: str, surface: str = 'Upper', frame: int = None, stat: str = 'mean'):

        '''
        Raw per-surfel values of Data/<surface>/<name>.

        Parameters
        ----------
        name : str
            Dataset name under Data/<surface> (spaces already converted to
            underscores by to_h5()/convert_snc_to_h5() - e.g.
            'Skin_Friction', 'static_pressure'). See self.available_variables
            if unsure what's in this file.
        frame : int or None
            int selects one frame's row directly (instantaneous) - `stat`
            is ignored in this case, since a statistic across frames isn't
            meaningful for a single one. None (default) reduces across
            every frame in the file via `stat`.
        stat : 'mean', 'rms', or 'raw_rms'
            Only used when frame is None.
            'mean': time-average - x_bar = mean(x, over frames).
            'rms': fluctuation RMS about the mean, i.e. the standard
            deviation - sqrt(mean((x - x_bar)^2)). This is what
            "Cp_rms"/unsteadiness conventionally means (how much a
            quantity fluctuates), not the raw signal amplitude.
            'raw_rms': sqrt(mean(x^2)) - includes any nonzero mean, so a
            quantity with a large steady component looks dominated by
            that rather than by its fluctuation. Rarely what you want;
            provided for completeness.

        Returns
        -------
        np.ndarray, shape (n_points,)
        '''

        with h5py.File(self.filename, 'r') as f:

            key = f'Data/{surface}/{name}'
            if key not in f:
                raise KeyError(
                    f"'{name}' not found under Data/{surface} of '{self.filename}' - "
                    f"available: {self.available_variables[surface]}"
                )

            if frame is not None:
                if not (0 <= frame < self.n_frames):
                    raise ValueError(f"frame={frame} out of range (n_frames={self.n_frames})")
                return f[key][frame]

            data = f[key][:]

        if stat == 'mean':
            return data.mean(axis=0)
        elif stat == 'rms':
            return np.sqrt(((data - data.mean(axis=0)) ** 2).mean(axis=0))
        elif stat == 'raw_rms':
            return np.sqrt((data ** 2).mean(axis=0))
        else:
            raise ValueError(f"Unknown stat '{stat}' - use 'mean', 'rms', or 'raw_rms'.")

    def cp(self, surface: str = 'Upper', frame: int = None, stat: str = 'mean',
           pressure_variable: str = 'Static_Pressure'):

        '''
        Pressure coefficient, Cp = (p - pref) / q_ref, with q_ref =
        0.5*rho_ref*(omega*r)^2 - the same LOCAL, per-surfel-radius
        normalization as FrictionLines.cf() (see README.md's "Equations"
        section), not one fixed reference velocity for the whole blade.

        For stat='rms'/'raw_rms': pref is not required and has no effect
        even if set - a constant offset doesn't change a fluctuation's
        standard deviation, so this reduces to variable(...)/q_ref.

        Parameters
        ----------
        pressure_variable : str
            Dataset name to treat as pressure - default matches
            SNCReader.to_h5()'s 'Static Pressure' -> 'Static_Pressure', but
            note that source is NOT validated for physical units (see
            SNCReader's docstring) - for real Cp, this should normally
            point at a pf2ens-derived file's pressure variable instead
            (e.g. 'static_pressure' from convert_snc_to_h5()).

        Returns
        -------
        np.ndarray, shape (n_points,)
        '''

        if self.rho_ref is None or self.rpm is None:
            raise ValueError("rho_ref and rpm must be set (in __init__) to compute Cp.")

        q_ref = self._q_ref(surface)

        if frame is None and stat == 'mean':
            if self.pref is None:
                raise ValueError("pref must be set (in __init__) to compute mean Cp.")
            p = self.variable(pressure_variable, surface=surface, frame=frame, stat='mean')
            return (p - self.pref) / q_ref

        if frame is not None:
            if self.pref is None:
                raise ValueError("pref must be set (in __init__) to compute instantaneous Cp.")
            p = self.variable(pressure_variable, surface=surface, frame=frame)
            return (p - self.pref) / q_ref

        p_stat = self.variable(pressure_variable, surface=surface, frame=None, stat=stat)
        return p_stat / q_ref

    def at_radii(self, radii, get_values, surface=('Upper', 'Lower'), tol: float = 0.0015,
                 n_chord_bins: int = 75, span_min: float = None, span_max: float = None,
                 chord_percentile: float = 0.1, edge_crop: float = 0.0, reverse_chord: bool = False):

        '''
        x/c-positioned values of an arbitrary per-surfel field, at each of
        several fixed radii, for one or both surfaces - generalizes
        FrictionLines.cf_at_radii() to any field via get_values, and to
        both surfaces at once.

        x/c is the raw chordwise Cartesian coordinate, normalized to
        [0, 1] using each (surface, radius band)'s own [chord_percentile,
        100-chord_percentile] percentile range (NOT literal min/max - see
        chord_percentile below) - a local shape check, not a
        case-independent x/c; see FrictionLines.cf_at_radii()'s docstring
        for the same caveat.

        Parameters
        ----------
        radii : array-like of float
            Target physical radii [m].
        get_values : callable(surface_label: str) -> np.ndarray
            Returns the per-surfel field to extract for 'Upper' or
            'Lower', shape (n_points,) matching that surface's own point
            count - e.g. lambda s: self.cp(surface=s, stat='rms').
        surface : str or tuple of str
            'Upper', 'Lower', or both (default).
        tol : float
            Half-width [m] of the radius band selected around each target.
        n_chord_bins : int or None
            x/c is binned into this many equal bins and the field is
            averaged within each (see cf_at_radii() for why - a thin
            radius band still contains a huge number of raw points, which
            reads as a noisy cloud otherwise). Default (75) is fewer/wider
            bins than FrictionLines.plot_cf_radii()'s default (150), i.e.
            deliberately smoother - Cp's mean field has less small-scale
            structure worth resolving than Cf's. Set to None to keep every
            raw surfel instead (no averaging at all).
        span_min, span_max : float, optional
            Keep only points with span_min <= span <= span_max (centered
            Cartesian span - see _span_chord()), BEFORE computing each
            radius band's chord min/max for x/c. Required in practice for
            any case with more than one blade sharing the same radius
            range (e.g. a 2+ bladed rotor where one face/file covers the
            whole rotor) - without it, a radius band picks up every
            blade's surfels at once, and x/c gets normalized against the
            combined chord range of ALL of them mixed together, which
            silently produces a garbled curve (validated: this is exactly
            what caused a spurious second peak in an early version of this
            plot - see friction_lines()'s identical span_min parameter and
            its docstring for the same issue in that class). No default
            here either, for the same reason FrictionLines doesn't have
            one: there's no reliable automatic value.
        chord_percentile : float
            x/c=0 and x/c=1 are anchored to the chord_percentile and
            (100-chord_percentile) percentiles of chord within each
            (surface, radius band), not the literal min/max - guards
            against a single extreme-chord outlier point stretching the
            whole normalization. Small by default (0.1); set to 0 to fall
            back to literal min/max.
        edge_crop : float
            Drop the outermost edge_crop fraction of x/c (after the
            chord_percentile normalization above) from each end of the
            OUTPUT, without moving where x/c=0/1 are. Off by default (0.0)
            - an earlier version of this docstring suggested using this to
            crop a "trailing edge anomaly" near x/c=1, but that turned out
            to be the genuine leading-edge stagnation point + suction peak
            (Cp approaching the theoretical stagnation value there, right
            next to the suction side's peak) sitting at the wrong end of
            an unflipped chord axis - see reverse_chord. Real physics, not
            an artifact to crop; kept as an option in case a genuine
            near-edge oddity shows up in some other case.
        reverse_chord : bool
            If True, flip which raw chord extreme maps to x/c=0 vs. x/c=1.
            The raw chord axis has no inherent leading/trailing-edge
            orientation - x/c=0 was arbitrarily assigned to the minimum
            chord value. Check which orientation is physically right per
            case: Cp should return to ~0 approaching the trailing edge
            (Kutta condition) and show a sharp stagnation-point/suction-
            peak feature (Cp near the theoretical stagnation value) right
            at the leading edge - if that signature is at x/c=1 instead of
            x/c=0, set this to True.

        Returns
        -------
        dict[float, dict[str, tuple[np.ndarray, np.ndarray]]]
            radius -> {surface_label: (xc, values)}, both sorted by xc (or
            bin-center order if n_chord_bins is set).
        '''

        surfaces = (surface,) if isinstance(surface, str) else tuple(surface)
        curves = {r_target: {} for r_target in radii}

        for surf in surfaces:

            r = self._radius(surf)
            span, chord = self._span_chord(surf)
            values = get_values(surf)

            span_mask = np.ones(len(span), dtype=bool)
            if span_min is not None:
                span_mask &= span >= span_min
            if span_max is not None:
                span_mask &= span <= span_max

            for r_target in radii:

                mask = span_mask & (np.abs(r - r_target) < tol)
                n_sel = int(mask.sum())

                if n_sel < 10:
                    raise ValueError(
                        f"Only {n_sel} points within {tol} m of r={r_target} on {surf} "
                        f"(after span_min/span_max cropping) - widen tol, check r_target, "
                        "or check span_min/span_max."
                    )

                c = chord[mask]
                v_masked = values[mask]

                # Percentile bounds, not literal min/max: guards against a
                # single extreme-chord outlier point stretching the whole
                # x/c normalization. Small by default (chord_percentile) -
                # NOT the fix for the near-x/c=1 pressure recovery some
                # cases show (see edge_crop below); that recovery is a
                # continuous feature of the raw data (no gap in chord
                # value, confirmed by inspection), not a single outlier
                # point, so no percentile choice removes it by itself.
                c_min, c_max = np.percentile(c, [chord_percentile, 100 - chord_percentile])
                valid = (c >= c_min) & (c <= c_max)
                xc_all = (c[valid] - c_min) / (c_max - c_min)
                if reverse_chord:
                    xc_all = 1 - xc_all
                v_valid = v_masked[valid]

                # edge_crop: drop the outermost edge_crop fraction of x/c on
                # each end from the OUTPUT (after normalizing against the
                # full range above, so this doesn't move where x/c=0/1 are -
                # it just hides that region). Use this for a near-TE (or
                # near-LE) region whose physical origin isn't pinned down -
                # e.g. a fast pressure recovery near x/c=1 close to the true
                # trailing edge, of uncertain origin (real blunt/rounded-TE
                # base pressure vs. residual Upper/Lower classification
                # ambiguity in a very thin region - both plausible, not
                # conclusively distinguished) - without silently discarding
                # data by redefining the x/c axis itself.
                edge_valid = (xc_all >= edge_crop) & (xc_all <= 1 - edge_crop)
                xc = xc_all[edge_valid]
                v_sel = v_valid[edge_valid]

                if n_chord_bins is None:
                    order = np.argsort(xc)
                    curves[r_target][surf] = (xc[order], v_sel[order])
                    continue

                bin_edges = np.linspace(0, 1, n_chord_bins + 1)
                bin_idx = np.clip(np.digitize(xc, bin_edges) - 1, 0, n_chord_bins - 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                v_mean = np.full(n_chord_bins, np.nan)
                counts = np.bincount(bin_idx, minlength=n_chord_bins)
                sums = np.bincount(bin_idx, weights=v_sel, minlength=n_chord_bins)
                has_data = counts > 0
                v_mean[has_data] = sums[has_data] / counts[has_data]

                curves[r_target][surf] = (bin_centers[has_data], v_mean[has_data])

        return curves

    def plot_at_radii(self, radii, get_values, ylabel: str, surface=('Upper', 'Lower'),
                       tol: float = 0.0015, n_chord_bins: int = 75, span_min: float = None,
                       span_max: float = None, chord_percentile: float = 0.1, edge_crop: float = 0.0,
                       reverse_chord: bool = False, marker_size: float = 12, cmap: str = 'cividis',
                       ax=None, savepath: str = None, dpi: int = 150):

        '''
        Plot of an arbitrary field vs local x/c at several radii, BOTH
        surfaces in the SAME color per radius (one legend entry each) -
        Upper and Lower each trace their own branch, colored by radius,
        not by surface.

        Always plotted as points (scatter), even with n_chord_bins set
        (bin-averaged, so already smooth): a connected line whose end sits
        wherever edge_crop or chord_percentile happened to cut it off
        visually reads as "the curve is missing/incomplete" past that
        point; discrete points don't imply the curve continues beyond
        them, so there's nothing to look "cut off".

        See at_radii() for what get_values/n_chord_bins/span_min/span_max/
        reverse_chord do - span_min/span_max in particular matters a lot
        here: without it, a case with more than one blade at the same
        radius (e.g. a 2-bladed rotor covered by one face/file) mixes
        every blade's chord into the same x/c range and produces a
        garbled curve.

        Returns
        -------
        (fig, ax)
        '''

        curves = self.at_radii(radii, get_values, surface=surface, tol=tol, n_chord_bins=n_chord_bins,
                                span_min=span_min, span_max=span_max, chord_percentile=chord_percentile,
                                edge_crop=edge_crop, reverse_chord=reverse_chord)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(radii)))

        for color, r_target in zip(colors, radii):
            first = True
            for surf, (xc, v) in curves[r_target].items():
                label = f'r={r_target:.3f} m' if first else None
                ax.scatter(xc, v, s=marker_size, color=color, label=label)
                first = False

        ax.set_xlabel(r'$x/c$ [-]')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax

    def plot_cp_radii(self, radii, frame: int = None, stat: str = 'mean',
                       pressure_variable: str = 'Static_Pressure', surface=('Upper', 'Lower'),
                       tol: float = 0.0015, n_chord_bins: int = 75, span_min: float = None,
                       span_max: float = None, chord_percentile: float = 0.1, edge_crop: float = 0.0,
                       reverse_chord: bool = False, marker_size: float = 12, cmap: str = 'cividis',
                       ax=None, savepath: str = None, dpi: int = 150):

        '''
        Convenience wrapper: plot_at_radii() for Cp (see cp()).

        frame : int or None
            A specific frame index for an instantaneous plot, or None
            (default) for the `stat`-reduced case (mean/rms/raw_rms over
            every frame in the file) - see cp()/variable(). Ignored if
            given together with stat='rms'/'raw_rms' (cp() itself ignores
            stat whenever frame is set - a statistic across frames isn't
            meaningful for a single one).
        stat : 'mean', 'rms', or 'raw_rms'
            Only used when frame is None - see variable(). 'mean' plots
            -Cp (negated - see below), ylabel '$-C_p$'; 'rms'/'raw_rms'
            plot Cp's rms/raw_rms directly (already non-negative,
            negating it would be wrong), ylabel '$C_{p,rms}$'. An
            instantaneous frame is also treated as 'mean'-like for the
            sign/ylabel choice, since it's a single Cp snapshot, not an
            rms statistic.

        -Cp, not Cp, whenever the ylabel is '$-C_p$': standard convention
        for this kind of plot, so the suction side's peak (Cp very
        negative there) reads as a peak going UP, not down - matches the
        LE stagnation-point dip (Cp~+1, so -Cp~-1) sitting right next to a
        sharp suction peak (-Cp~+1.2-1.3) on the correct (suction) branch,
        and a flatter pressure-side branch, once actually negated.

        NOTE: BladePostProcessor.plot_radii() (elsewhere in this project)
        labels its y-axis '$-C_p$' the same way but does NOT actually
        negate the curve it plots - confirmed by reading that method
        directly. Not fixed here (different class, wasn't asked to), but
        worth knowing if comparing against or reusing that tool's output
        or past figures - they're plotting +Cp under a -Cp label.
        '''

        is_rms = frame is None and stat in ('rms', 'raw_rms')
        sign = 1 if is_rms else -1
        get_values = lambda s: sign * self.cp(surface=s, frame=frame, stat=stat,
                                               pressure_variable=pressure_variable)
        ylabel = r'$C_{p,rms}$ [-]' if is_rms else r'$-C_p$ [-]'

        return self.plot_at_radii(radii, get_values, ylabel, surface=surface, tol=tol,
                                   n_chord_bins=n_chord_bins, span_min=span_min, span_max=span_max,
                                   chord_percentile=chord_percentile, edge_crop=edge_crop,
                                   reverse_chord=reverse_chord, marker_size=marker_size, cmap=cmap, ax=ax,
                                   savepath=savepath, dpi=dpi)

    def pressure_fluctuation(self, frame: int, surface: str = 'Upper',
                              pressure_variable: str = 'Static_Pressure'):

        '''
        Instantaneous pressure fluctuation, p'(frame) = p(frame) - p_mean,
        where p_mean is the mean over EVERY frame in the file (not just up
        to `frame`) - the same mean variable(stat='mean')/'rms' use, so
        this is consistent with Prms = sqrt(mean(p'^2)) (variable(...,
        stat='rms')) rather than some other baseline.

        Dimensional [Pa], not normalized by q_ref - unlike cp(), since a
        fluctuation-about-the-mean's sign/shape is the point here, not a
        comparison across radii with very different q_ref.

        Parameters
        ----------
        frame : int
            Which frame's instantaneous snapshot to compute the
            fluctuation for.
        pressure_variable : str
            Dataset name to treat as pressure - see cp()'s parameter of
            the same name.

        Returns
        -------
        np.ndarray, shape (n_points,)
        '''

        p_frame = self.variable(pressure_variable, surface=surface, frame=frame)
        p_mean = self.variable(pressure_variable, surface=surface, frame=None, stat='mean')

        return p_frame - p_mean

    def plot_pressure_fluctuation(self, frame: int, surface=('Upper', 'Lower'),
                                   pressure_variable: str = 'Static_Pressure', span_min: float = None,
                                   span_max: float = None, value_clip_percentile: float = 99,
                                   marker_size: float = 1, cmap: str = 'cividis', figsize: tuple = None,
                                   savepath: str = None, dpi: int = 150):

        '''
        Convenience wrapper: plot_variable_surface() for one frame's
        pressure_fluctuation() (see that method) - one blade contour per
        frame, call once per frame of interest (e.g. in a loop over
        range(sv.n_frames) to build an animation's frames).

        Returns
        -------
        (fig, axes)
        '''

        get_values = lambda s: self.pressure_fluctuation(frame, surface=s,
                                                           pressure_variable=pressure_variable)

        return self.plot_variable_surface(get_values, cbar_label=r"$p'$ [Pa]", surface=surface,
                                           span_min=span_min, span_max=span_max,
                                           value_clip_percentile=value_clip_percentile,
                                           marker_size=marker_size, cmap=cmap, figsize=figsize,
                                           savepath=savepath, dpi=dpi)

    def _nearest_point(self, span_pct: float, chord_pct: float, surface: str = 'Upper',
                        tol: float = 0.0015, chord_percentile: float = 0.1, reverse_chord: bool = False,
                        span_min: float = None, span_max: float = None):

        '''
        Index of the raw surfel nearest a target (r/R, x/c) location,
        given as percentages (0-100 each) - r/R uses this instance's
        r_tip (constructor arg), x/c uses the same local, per-radius-band
        percentile-normalized chord convention as at_radii()/
        to_common_grid() (see chord_percentile/reverse_chord there for
        what these mean and how to check which orientation is right).

        span_min/span_max matter here for the same reason as everywhere
        else in this class (see friction_lines()/at_radii()): without
        them, on a multi-blade file the nearest point in x/c could come
        from the wrong blade.

        Returns
        -------
        idx : int
            Index into this surface's raw point arrays (self.positions[surface],
            any Data/<surface>/<name> row/column).
        r_actual, xc_actual : float
            The ACTUAL (radius [m], chord fraction) of the selected point
            - report these alongside any requested span_pct/chord_pct,
            since the nearest raw surfel generally won't sit exactly on
            the target.
        '''

        if self.r_tip is None:
            raise ValueError("r_tip must be set (in __init__) to locate a point by span percentage.")

        r_target = (span_pct / 100.0) * self.r_tip
        xc_target = chord_pct / 100.0

        r = self._radius(surface)
        span, chord = self._span_chord(surface)

        mask = np.ones(len(r), dtype=bool)
        if span_min is not None:
            mask &= span >= span_min
        if span_max is not None:
            mask &= span <= span_max

        band = mask & (np.abs(r - r_target) < tol)
        n_sel = int(band.sum())
        if n_sel < 10:
            raise ValueError(
                f"Only {n_sel} points within {tol} m of r={r_target:.4f} m (span_pct={span_pct}) "
                f"on {surface} (after span_min/span_max cropping) - widen tol, check span_pct, "
                "or check span_min/span_max."
            )

        idx_band = np.flatnonzero(band)
        c = chord[band]
        c_min, c_max = np.percentile(c, [chord_percentile, 100 - chord_percentile])
        xc = (c - c_min) / (c_max - c_min)
        if reverse_chord:
            xc = 1 - xc

        local_idx = int(np.argmin(np.abs(xc - xc_target)))
        idx = int(idx_band[local_idx])

        return idx, float(r[idx]), float(xc[local_idx])

    def timetrace(self, name: str, span_pct: float, chord_pct: float, surface: str = 'Upper',
                  tol: float = 0.0015, chord_percentile: float = 0.1, reverse_chord: bool = False,
                  span_min: float = None, span_max: float = None, dt: float = None):

        '''
        Time trace of Data/<surface>/<name> at ONE raw surfel nearest the
        given (span_pct, chord_pct) location (see _nearest_point()) -
        every frame's value at that single point, e.g. for a Welch
        periodogram of wall pressure fluctuations (see periodogram()).

        Parameters
        ----------
        name : str
            Dataset name under Data/<surface> - see variable().
        span_pct, chord_pct : float
            Target location as percentages (0-100) of r/R and x/c - see
            _nearest_point() for exactly how these map to a raw point
            (the nearest available surfel, NOT an interpolated value).
        dt : float, optional
            Physical timestep [s] between consecutive frames, for the
            returned time axis. If not given, falls back to
            Metadata/mid_s (real per-frame time in seconds - only
            present in convert_snc_to_h5()-produced files built with an
            nc_stats file, see _load()) if available, else the returned
            time axis is just the integer frame index (dimensionless) -
            fine for inspecting a trace's shape, but periodogram() needs
            a real sampling rate, so pass dt there (or here, doesn't
            matter) if neither of these is available.

        Returns
        -------
        t : np.ndarray, shape (n_frames,)
            Time [s] if dt or Metadata/mid_s is available, else the raw
            frame index.
        values : np.ndarray, shape (n_frames,)
        point_info : dict
            {'idx', 'r', 'xc', 'surface'} - the ACTUAL point used (see
            _nearest_point()), since the nearest raw surfel won't sit
            exactly on the requested span_pct/chord_pct.
        '''

        idx, r_actual, xc_actual = self._nearest_point(
            span_pct, chord_pct, surface=surface, tol=tol, chord_percentile=chord_percentile,
            reverse_chord=reverse_chord, span_min=span_min, span_max=span_max)

        with h5py.File(self.filename, 'r') as f:

            key = f'Data/{surface}/{name}'
            if key not in f:
                raise KeyError(
                    f"'{name}' not found under Data/{surface} of '{self.filename}' - "
                    f"available: {self.available_variables[surface]}"
                )
            values = f[key][:, idx]

        if dt is not None:
            t = np.arange(self.n_frames) * dt
        elif self.frame_times is not None:
            t = self.frame_times
        else:
            t = np.arange(self.n_frames)

        point_info = {'idx': idx, 'r': r_actual, 'xc': xc_actual, 'surface': surface}

        return t, values, point_info

    def plot_timetrace(self, name: str, span_pct: float, chord_pct: float, surface: str = 'Upper',
                        tol: float = 0.0015, chord_percentile: float = 0.1, reverse_chord: bool = False,
                        span_min: float = None, span_max: float = None, dt: float = None,
                        ylabel: str = None, ax=None, savepath: str = None, dpi: int = 150):

        '''
        Plot timetrace() as a connected line vs time (or frame index if
        neither dt nor Metadata/mid_s is available) - a genuine line
        plot, unlike the scatter convention used elsewhere in this class
        (at_radii()/plot_variable_surface()): time is monotonic with
        exactly one sample per frame, so connecting consecutive samples
        IS the signal, not an interpolation artifact papering over a
        cropped/percentile-cut end (see plot_at_radii()'s docstring for
        why THAT case is different).

        Returns
        -------
        (fig, ax)
        '''

        t, values, point_info = self.timetrace(
            name, span_pct, chord_pct, surface=surface, tol=tol, chord_percentile=chord_percentile,
            reverse_chord=reverse_chord, span_min=span_min, span_max=span_max, dt=dt)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.figure

        has_time = dt is not None or self.frame_times is not None

        ax.plot(t, values, color='k', linewidth=1)
        ax.set_xlabel('Time [s]' if has_time else 'Frame index')
        ax.set_ylabel(ylabel or name)
        ax.set_title(f"{point_info['surface']}, $r/R={point_info['r'] / self.r_tip:.3f}$, "
                     f"$x/c={point_info['xc']:.3f}$")
        ax.grid(True)
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax

    def periodogram(self, name: str, span_pct: float, chord_pct: float, surface: str = 'Upper',
                     fs: float = None, dt: float = None, nperseg: int = None, detrend='constant',
                     tol: float = 0.0015, chord_percentile: float = 0.1, reverse_chord: bool = False,
                     span_min: float = None, span_max: float = None, **welch_kwargs):

        '''
        Welch power spectral density of timetrace() at one point - wraps
        scipy.signal.welch, e.g. for a wall pressure fluctuation
        spectrum.

        Parameters
        ----------
        fs : float, optional
            Sampling frequency [Hz]. If not given, derived from dt
            (1/dt) or from Metadata/mid_s (1/mean(diff(mid_s)) - see
            timetrace()'s dt parameter for where that comes from). One
            of fs, dt, or a usable Metadata/mid_s is REQUIRED - unlike
            timetrace(), which tolerates a frame-index-only time axis
            for just inspecting a trace's shape, a periodogram's
            frequency axis is meaningless without a real sampling rate.
        nperseg : int, optional
            Passed to scipy.signal.welch - length of each Welch segment.
            Defaults to min(256, n_frames) rather than scipy's own
            default (a bare 256): scipy would silently clip nperseg down
            to n_frames anyway if it's larger, but doing that explicitly
            here avoids a confusing implicit default when n_frames is
            still short (a likely case while more frames are still being
            generated - see this method's motivating use case).
        detrend : str or False
            Passed to scipy.signal.welch - 'constant' (default) removes
            each segment's mean before its FFT, appropriate for a raw
            (non-mean-removed) signal like pressure itself. Set to False
            if plotting an already mean-removed signal (e.g.
            pressure_fluctuation() sampled over frames) to avoid
            double-removing the mean.
        **welch_kwargs
            Any other scipy.signal.welch keyword (window, noverlap, ...).

        Returns
        -------
        freq : np.ndarray, [Hz]
        psd : np.ndarray
            Power spectral density, units = (value units)^2 / Hz.
        point_info : dict
            See timetrace().
        '''

        t, values, point_info = self.timetrace(
            name, span_pct, chord_pct, surface=surface, tol=tol, chord_percentile=chord_percentile,
            reverse_chord=reverse_chord, span_min=span_min, span_max=span_max, dt=dt)

        if fs is None:
            if dt is not None:
                fs = 1.0 / dt
            elif self.frame_times is not None:
                fs = 1.0 / np.mean(np.diff(self.frame_times))
            else:
                raise ValueError(
                    "fs (or dt, or a usable Metadata/mid_s) is required to compute a periodogram - "
                    "a frame-index-only time axis has no physical sampling rate. See timetrace()'s "
                    "dt parameter for why."
                )

        if nperseg is None:
            nperseg = min(256, len(values))

        freq, psd = welch(values, fs=fs, nperseg=nperseg, detrend=detrend, **welch_kwargs)

        return freq, psd, point_info

    def plot_periodogram(self, name: str, span_pct: float, chord_pct: float, surface: str = 'Upper',
                          fs: float = None, dt: float = None, nperseg: int = None, detrend='constant',
                          tol: float = 0.0015, chord_percentile: float = 0.1, reverse_chord: bool = False,
                          span_min: float = None, span_max: float = None, ylabel: str = None, ax=None,
                          savepath: str = None, dpi: int = 150, **welch_kwargs):

        '''
        Plot periodogram() as log-log PSD vs frequency - the standard way
        to read a spectrum's decades of magnitude/frequency together; a
        linear axis flattens everything below the largest peak into an
        indistinguishable baseline.

        Returns
        -------
        (fig, ax)
        '''

        freq, psd, point_info = self.periodogram(
            name, span_pct, chord_pct, surface=surface, fs=fs, dt=dt, nperseg=nperseg, detrend=detrend,
            tol=tol, chord_percentile=chord_percentile, reverse_chord=reverse_chord, span_min=span_min,
            span_max=span_max, **welch_kwargs)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        # Skip freq=0 (DC) - undefined on a log axis, and not a frequency
        # component anyway (that's what detrend='constant' already removes).
        start = 1 if freq[0] == 0 else 0

        ax.loglog(freq[start:], psd[start:], color='k', linewidth=1.2)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(ylabel or 'PSD')
        ax.set_title(f"{point_info['surface']}, $r/R={point_info['r'] / self.r_tip:.3f}$, "
                     f"$x/c={point_info['xc']:.3f}$")
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax

    def plot_variable_surface(self, get_values, cbar_label: str, surface=('Upper', 'Lower'),
                               get_vector=None, span_min: float = None, span_max: float = None,
                               value_clip_percentile: float = 99, n_arrows: int = 2000,
                               marker_size: float = 1, cmap: str = 'cividis', figsize: tuple = None,
                               savepath: str = None, dpi: int = 150):

        '''
        Scatter of an arbitrary scalar field over the blade surface (raw
        Cartesian span/chord - see class docstring), one row per surface
        (Upper/Lower by default, stacked) - generalizes
        FrictionLines.friction_lines() beyond Cf to any field (static
        pressure, Cp, y+, ...) via get_values, the same pattern
        at_radii()/plot_at_radii() already use.

        Deliberately NOT interpolated onto a grid first - see
        FrictionLines.friction_lines()'s docstring for why (an earlier
        version of that method did, and it was both slow and introduced
        interpolation artifacts). Plotting every surfel directly as a
        small point already reads as a continuous field.

        Parameters
        ----------
        get_values : callable(surface_label: str) -> np.ndarray
            Returns the per-surfel field to color by, e.g.
            lambda s: self.cp(surface=s, stat='mean').
        get_vector : callable(surface_label: str) -> (span_component, chord_component), optional
            If given, overlays a sparse quiver of this vector field's
            direction (e.g. surface velocity) - same style as
            FrictionLines.friction_lines()'s wall-shear quiver. Both
            components must be np.ndarray, shape matching get_values'
            output for that surface.
        span_min, span_max : float, optional
            Keep only points with span_min <= span <= span_max (centered
            Cartesian span - see _span_chord()) - matters for any case
            with more than one blade at the same span range (see
            FrictionLines.friction_lines()'s docstring for the same
            issue); no reliable automatic value, pass what's right for
            this case's mesh.
        value_clip_percentile : float
            Colors are clipped to the [100-value_clip_percentile,
            value_clip_percentile] percentile range of the field before
            plotting, since without clipping a handful of extreme surfels
            wash out the entire color scale. Two-sided (unlike
            FrictionLines.friction_lines()'s [0, vmax], which assumes a
            non-negative magnitude) - a general field like Cp can be
            negative. Set to 100 to disable.
        n_arrows : int
            Number of surfels randomly sampled for the direction quiver
            (only used if get_vector is given).
        marker_size : float
            Scatter marker size (matplotlib's `s`, points^2).

        Returns
        -------
        (fig, axes)
            axes is a single Axes if surface is a single string, otherwise
            a list of Axes in the same order as surface.
        '''

        surfaces = (surface,) if isinstance(surface, str) else tuple(surface)
        rng = np.random.default_rng(0)

        if figsize is None:
            figsize = (14, 4.5 * len(surfaces))

        fig, axes = plt.subplots(len(surfaces), 1, figsize=figsize, sharex=True, squeeze=False)
        axes = axes[:, 0]

        for ax, surf in zip(axes, surfaces):

            span, chord = self._span_chord(surf)
            values = get_values(surf)

            mask = np.ones(len(span), dtype=bool)
            if span_min is not None:
                mask &= span >= span_min
            if span_max is not None:
                mask &= span <= span_max

            span_sel, chord_sel, v_sel = span[mask], chord[mask], values[mask]

            vmin, vmax = np.percentile(v_sel, [100 - value_clip_percentile, value_clip_percentile])
            v_clipped = np.clip(v_sel, vmin, vmax)

            sc = ax.scatter(span_sel, chord_sel, c=v_clipped, s=marker_size, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(cbar_label)

            if get_vector is not None:
                vec_span, vec_chord = get_vector(surf)
                vec_span_sel, vec_chord_sel = vec_span[mask], vec_chord[mask]
                idx = rng.choice(len(span_sel), size=min(n_arrows, len(span_sel)), replace=False)
                mag = np.hypot(vec_span_sel[idx], vec_chord_sel[idx])
                mag[mag == 0] = 1
                ax.quiver(span_sel[idx], chord_sel[idx], vec_span_sel[idx] / mag, vec_chord_sel[idx] / mag,
                          color='white', scale=60, width=0.002, alpha=0.8)

            ax.set_ylabel('chord [m]')
            ax.set_aspect('equal')

        axes[-1].set_xlabel('span [m]')
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, (axes[0] if len(axes) == 1 else list(axes))

    def to_common_grid(self, get_values, r_target, x_target, surface: str = 'Upper',
                        span_min: float = None, span_max: float = None, n_radius_bins: int = 100,
                        chord_percentile: float = 0.1, reverse_chord: bool = False):

        '''
        Resample an arbitrary per-surfel field onto a shared (r/R, x/c)
        grid, so it can be compared/differenced against another case with
        a different native mesh - same purpose and grid convention as
        SurfaceField.to_common_grid() (bladeprocessor/surface_field.py),
        built directly from this raw surfel cloud instead of a
        pre-resampled (Radius, Chord) file. See field(), which wraps this
        as a SurfaceField-compatible object for direct use with
        SurfaceFieldComparator.

        The blade span (after span_min/span_max cropping) is first
        partitioned into n_radius_bins equal-width radius bins, each
        independently resampled onto x_target (same per-band
        chord_percentile/reverse_chord handling as at_radii() - see that
        method's docstring for what these mean and why), THEN interpolated
        across bins onto r_target - the same two-stage approach
        SurfaceField.to_common_grid() uses, just building the first stage
        from scattered raw points instead of an already-native (radius,
        chord) grid.

        Parameters
        ----------
        get_values : callable(surface_label: str) -> np.ndarray
        r_target : array-like
            Target r/R values, increasing, in [0, 1] - uses this
            instance's r_tip (constructor arg) for R.
        x_target : array-like
            Target x/c values, increasing, in [0, 1].
        n_radius_bins : int
            Number of radius bins the raw span is partitioned into before
            interpolating onto r_target - resolution of the "native" grid
            this resamples from. Bins with fewer than 10 points are left
            as NaN (skipped, matching SurfaceField's own handling of
            missing rows/columns).

        Returns
        -------
        np.ndarray, shape (len(r_target), len(x_target))
            Resampled field, NaN outside the covered region.
        '''

        if self.r_tip is None:
            raise ValueError("r_tip must be set (in __init__) to compute r/R for a common grid.")

        r = self._radius(surface)
        _, chord = self._span_chord(surface)
        values = get_values(surface)

        mask = np.ones(len(r), dtype=bool)
        if span_min is not None:
            span, _ = self._span_chord(surface)
            mask &= span >= span_min
        if span_max is not None:
            span, _ = self._span_chord(surface)
            mask &= span <= span_max

        r_m, chord_m, v_m = r[mask], chord[mask], values[mask]

        bin_edges = np.linspace(r_m.min(), r_m.max(), n_radius_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        chordwise = np.full((n_radius_bins, len(x_target)), np.nan)

        for i in range(n_radius_bins):

            bin_mask = (r_m >= bin_edges[i]) & (r_m < bin_edges[i + 1] if i < n_radius_bins - 1
                                                 else r_m <= bin_edges[i + 1])
            if bin_mask.sum() < 10:
                continue

            c, v = chord_m[bin_mask], v_m[bin_mask]
            c_min, c_max = np.percentile(c, [chord_percentile, 100 - chord_percentile])
            valid = (c >= c_min) & (c <= c_max)
            if valid.sum() < 2:
                continue

            xc = (c[valid] - c_min) / (c_max - c_min)
            if reverse_chord:
                xc = 1 - xc

            order = np.argsort(xc)
            chordwise[i] = np.interp(x_target, xc[order], v[valid][order], left=np.nan, right=np.nan)

        r_over_R = bin_centers / self.r_tip

        out = np.full((len(r_target), len(x_target)), np.nan)
        row_has_data = np.sum(~np.isnan(chordwise), axis=1) > 0
        order = np.argsort(r_over_R[row_has_data])
        r_valid = r_over_R[row_has_data][order]
        chordwise_valid = chordwise[row_has_data][order]

        for j in range(len(x_target)):
            col = chordwise_valid[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                continue
            out[:, j] = np.interp(r_target, r_valid[valid], col[valid], left=np.nan, right=np.nan)

        return out

    def field(self, get_values, var_name: str, c_ref: float = None, surface: str = 'Upper',
              **to_common_grid_kwargs):

        '''
        Wrap (this instance, get_values) as a SurfaceVariableField -
        drop-in compatible with SurfaceFieldComparator
        (bladeprocessor/surface_field.py), unmodified, for delta/side-by-
        side comparison against either another SurfaceVariableField or an
        actual SurfaceField (they share the same (r/R, x/c) grid
        convention).

        Parameters
        ----------
        get_values : callable(surface_label: str) -> np.ndarray
        var_name : str
            Label used for titles/colorbars if not overridden explicitly.
        c_ref : float, optional
            Reference chord length [m] for x/c normalization - required
            (raises when actually used, not here) since raw surfel data
            has no single obvious "reference chord" the way a
            pre-resampled (Radius, Chord) file's own Chord axis provides
            one; pass the same physical value for every case being
            compared, or x/c won't mean the same thing across them.
        **to_common_grid_kwargs
            Forwarded to to_common_grid() on every call (span_min,
            span_max, n_radius_bins, chord_percentile, reverse_chord).

        Returns
        -------
        SurfaceVariableField
        '''

        return SurfaceVariableField(self, get_values, var_name, c_ref=c_ref, surface=surface,
                                     **to_common_grid_kwargs)


class SurfaceVariableField:

    '''
    Binds a (SurfaceVariable, get_values) pair as a single resampled
    field - duck-types SurfaceField's interface (to_common_grid(),
    physical_aspect(), var_name) so it can be passed directly into
    SurfaceFieldComparator (bladeprocessor/surface_field.py) unmodified.
    Build one via SurfaceVariable.field(), not directly.
    '''

    def __init__(self, surface_variable: 'SurfaceVariable', get_values, var_name: str,
                 c_ref: float = None, surface: str = 'Upper', **to_common_grid_kwargs):

        self.sv = surface_variable
        self.get_values = get_values
        self.var_name = var_name
        self.c_ref = c_ref
        self.r_tip = surface_variable.r_tip
        self.surface = surface
        self.to_common_grid_kwargs = to_common_grid_kwargs

    def chord_ref(self):

        '''Reference chord length for x/c normalization - see field()'s c_ref parameter.'''

        if self.c_ref is None:
            raise ValueError(
                f"'{self.var_name}': c_ref must be provided (SurfaceVariable.field(..., c_ref=...)) "
                "to compute x/c - unlike SurfaceField, raw surfel data has no native chord axis to "
                "fall back on."
            )
        return self.c_ref

    def physical_aspect(self):

        '''Physical aspect ratio (c_ref / r_tip) - see SurfaceField.physical_aspect().'''

        return self.chord_ref() / self.r_tip

    def to_common_grid(self, r_target, x_target, surface: str = None):

        '''Resample onto the shared grid - see SurfaceVariable.to_common_grid().'''

        surface = surface or self.surface
        return self.sv.to_common_grid(self.get_values, r_target, x_target, surface=surface,
                                       **self.to_common_grid_kwargs)
