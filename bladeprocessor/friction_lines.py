import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})


class FrictionLines:

    '''
    Wall shear stress / friction line post-processing from a SNCReader-
    derived HDF5 file (converters.snc_reader.SNCReader.to_h5(...,
    surface_split=True)) - NOT the pf2ens/pressure branch, whose
    EnSight-recomputed normals aren't trustworthy for this (see README.md,
    "Splitting into upper/lower surface").

    Instantaneous vs. average
    --------------------------
    Every method that touches the force field takes a `frame` argument:
    an int selects one frame (instantaneous), None (default) averages the
    force field over every frame stored in the file first (the "average"
    case). Passing an already PowerFLOW-side time-averaged .snc file
    (n_frames == 1) with frame=None works the same way - there's nothing
    case-specific to configure, since to_h5() always writes every frame
    present as a row, whether that's 1 (pre-averaged) or many (raw).

    Span/chord/thickness axes
    --------------------------
    Chordwise and spanwise position (used by cf_at_radii()/friction_lines())
    come straight from this file's raw Cartesian Geometry/X,Y,Z, selected
    via span_axis/chord_axis/thickness_axis (default 0/2/1, i.e. X/Z/Y) -
    NOT derived from the rotor's rotation axis. An earlier version of this
    class tried the latter (projecting position onto directions built from
    lrf_axis_direction, treating "along the rotation axis" as pure
    thickness and "in the disk plane" as span+chord) and it broke down on
    a real blade: any real geometric pitch/twist tilts the true chord line
    out of the disk plane, and no amount of re-binning fixed the resulting
    distortion (verified: per-radius-band chord width came out
    inconsistent and much larger than the known reference chord, worse
    with more bins, not better). Falling back to the mesh's own raw
    Cartesian axes sidesteps needing to reconstruct that pitch/twist at
    all - it works as long as this case's mesh was built with span/chord/
    thickness aligned to X/Y/Z (true for every case in this project so
    far), and the axis indices are there to override if a differently
    -oriented mesh ever shows up.

    The ROTATION axis (lrf_axis_direction) is still used for one thing:
    the physical radius r in cf_at_radii(), since "radius" is a rotor
    quantity that has to come from the rotation axis, not from Cartesian
    position alone - that part was validated independently (r came out
    0-0.1256 m for a known 0.125 m tip radius) and isn't affected by the
    chordwise-axis issue above.

    Cf normalization
    -----------------
    Cf = tau / q_ref, with q_ref = 0.5 * rho_ref * (omega * r)^2 - a LOCAL
    dynamic pressure, using each surfel's own radius r (from the rotation
    axis - see _radius()), not one fixed reference velocity for the whole
    blade. This matches BladePostProcessor.compute_cf() elsewhere in this
    project (same formula, U_ref(r) = omega * r), rather than
    non-dimensionalizing by a single global freestream/tip velocity - see
    README.md's "Equations" section for the full derivation and how this
    compares to the alternative.

    Parameters
    ----------
    filename : str
        Path to a SNCReader.to_h5(..., surface_split=True) HDF5 file.
    r_tip : float, optional
        Physical tip radius [m], for reference only - not required for
        anything below to run.
    rho_ref, rpm : float, optional
        Reference density [kg/m^3] and rotor speed [rev/min] for
        Cf = tau / (0.5 rho (omega r)^2) - see "Cf normalization" above.
        Required by cf() / cf_at_radii() / plot_cf_radii() /
        friction_lines(), not by wall_shear() (dimensional wall shear).
    span_axis, chord_axis, thickness_axis : int
        Which raw position column (0=X, 1=Y, 2=Z) is spanwise, chordwise,
        and thickness-wise for this case's mesh. Defaults (0, 2, 1) match
        every case seen in this project so far - override if a
        differently-oriented mesh ever comes up.
    '''

    def __init__(self, filename: str, r_tip: float = None, rho_ref: float = None, rpm: float = None,
                 span_axis: int = 0, chord_axis: int = 2, thickness_axis: int = 1):

        self.filename = filename
        self.r_tip = r_tip
        self.rho_ref = rho_ref
        self.rpm = rpm
        self.span_axis = span_axis
        self.chord_axis = chord_axis
        self.thickness_axis = thickness_axis
        self._load()

    def _load(self):

        '''Load geometry, normals and the per-frame force field for both surfaces.'''

        with h5py.File(self.filename, 'r') as f:

            if 'Upper' not in f['Geometry'] or 'Lower' not in f['Geometry']:
                raise ValueError(
                    f"'{self.filename}' has no Geometry/Upper or Geometry/Lower group - "
                    "was it written with SNCReader.to_h5(..., surface_split=True)?"
                )

            axis_origin = f['Metadata/lrf_axis_origin'][:]
            axis_direction = f['Metadata/lrf_axis_direction'][:]
            self.axis_origin = axis_origin
            self.axis_direction = axis_direction / np.linalg.norm(axis_direction)
            self.n_frames = f['Metadata/frame_index'].shape[0]

            self.surfaces = {}
            for label in ('Upper', 'Lower'):

                geo = f[f'Geometry/{label}']
                data = f[f'Data/{label}']

                self.surfaces[label] = {
                    'positions': np.column_stack([geo['X'][:], geo['Y'][:], geo['Z'][:]]),
                    'normals': np.column_stack([geo['NX'][:], geo['Normal_Y'][:], geo['Normal_Z'][:]]),
                    'force': np.stack([
                        data['Surface_X-Force'][:],
                        data['Surface_Y-Force'][:],
                        data['Surface_Z-Force'][:],
                    ], axis=-1),  # shape (n_frames, n_points, 3)
                }

    def _radius(self, surface: str):

        '''
        Physical radius from the rotor's rotation axis, for every surfel -
        the one place this class still uses lrf_axis_direction (see class
        docstring).
        '''

        positions = self.surfaces[surface]['positions']
        rel = positions - self.axis_origin
        along = rel @ self.axis_direction
        radial_vec = rel - along[:, None] * self.axis_direction

        return np.linalg.norm(radial_vec, axis=1)

    def _q_ref(self, surface: str):

        '''
        Local dynamic pressure q_ref = 0.5 * rho_ref * (omega * r)^2 at
        every surfel, r being this surfel's own radius (_radius()) - see
        "Cf normalization" in the class docstring.

        WARNING: q_ref -> 0 as r -> 0, so Cf blows up near the rotation
        axis. Some faces (e.g. this project's "Rotor::Default-Segment")
        include a bit of hub/bore geometry very close to r=0 - exclude it
        before computing Cf over a full, unrestricted selection (e.g. via
        span_min/span_max in friction_lines(), or keep radii away from ~0
        in cf_at_radii(), which already only ever looks at thin bands
        around radii you choose).
        '''

        omega = self.rpm * 2 * np.pi / 60
        r = self._radius(surface)

        return 0.5 * self.rho_ref * (omega * r) ** 2

    def _span_chord(self, surface: str):

        '''
        Raw Cartesian (span, chord) position for every surfel, each
        centered on this surface's own bounds (matches the convention
        already validated in this project's friction_lines_normal_split.py
        - see class docstring for why this is Cartesian, not
        rotation-axis-derived).
        '''

        positions = self.surfaces[surface]['positions']
        span = positions[:, self.span_axis]
        chord = positions[:, self.chord_axis]
        span = span - (span.min() + span.max()) / 2
        chord = chord - (chord.min() + chord.max()) / 2

        return span, chord

    def _force(self, surface: str, frame: int = None):

        '''Force vector per surfel: one frame, or the mean over all frames if frame is None.'''

        F = self.surfaces[surface]['force']

        if frame is not None:
            if not (0 <= frame < self.n_frames):
                raise ValueError(f"frame={frame} out of range for '{self.filename}' (n_frames={self.n_frames})")
            return F[frame]

        return F.mean(axis=0)

    def wall_shear(self, surface: str = 'Upper', frame: int = None):

        '''
        Wall shear vector tau = F - (F.n)n at every surfel, n being this
        surfel's own normal (from the raw .snc, not pf2ens's).

        Parameters
        ----------
        surface : 'Upper' or 'Lower'
        frame : int or None
            A specific frame index for an instantaneous result, or None
            (default) for the average case - see class docstring.

        Returns
        -------
        np.ndarray, shape (n_points, 3)
        '''

        F = self._force(surface, frame)
        normals = self.surfaces[surface]['normals']
        f_normal = np.sum(F * normals, axis=1)

        return F - f_normal[:, None] * normals

    def cf(self, surface: str = 'Upper', frame: int = None, component: str = None):

        '''
        Skin friction coefficient at every surfel, Cf = tau / q_ref, with
        q_ref = 0.5 * rho_ref * (omega * r)^2 - a LOCAL dynamic pressure
        using each surfel's own radius (see "Cf normalization" in the
        class docstring), not one fixed velocity for the whole blade.

        Parameters
        ----------
        component : None, 'chordwise', or 'spanwise'
            None (default): magnitude |tau| / q_ref - always >= 0, cannot
            reveal a change in flow state (e.g. separation/reattachment),
            since a sign reversal in the underlying vector just becomes a
            dip toward zero, not a crossing.
            'chordwise': signed tau[chord_axis] / q_ref - crosses zero
            exactly where the near-wall flow reverses direction along the
            chord, which is the actual separation/reattachment signal.
            'spanwise': signed tau[span_axis] / q_ref - crosses zero where
            near-wall flow reverses along the span (e.g. centrifugal
            pumping vs. inward migration).

        Returns
        -------
        np.ndarray, shape (n_points,)
        '''

        if self.rho_ref is None or self.rpm is None:
            raise ValueError("rho_ref and rpm must be set (in __init__) to compute Cf.")

        tau = self.wall_shear(surface=surface, frame=frame)
        q_ref = self._q_ref(surface)

        if component is None:
            value = np.linalg.norm(tau, axis=1)
        elif component == 'chordwise':
            value = tau[:, self.chord_axis]
        elif component == 'spanwise':
            value = tau[:, self.span_axis]
        else:
            raise ValueError(f"Unknown component '{component}' - use None, 'chordwise', or 'spanwise'.")

        return value / q_ref

    def cf_at_radii(self, radii, surface: str = 'Upper', frame: int = None, component: str = None,
                     tol: float = 0.0015, n_chord_bins: int = 150, span_min: float = None,
                     span_max: float = None, reverse_chord: bool = False):

        '''
        Cf vs. local chordwise position, at each of several fixed radii,
        ready to plot together (see plot_cf_radii()).

        x/c here is the raw chordwise Cartesian coordinate (see class
        docstring), normalized to [0, 1] using THIS BAND's own min/max -
        a local shape check, not a case-independent x/c: two bands at
        different radii are each independently rescaled, so a feature at
        "x/c=0.3" in one band isn't guaranteed to be at the same physical
        fraction of the chord in another. A proper global (r/R, x/c)
        resampling of raw .snc surfel clouds - shared with SurfaceField's
        grid layout - is still an open item (see README.md, "What's still
        open"); this is a narrower, already-useful stand-in for looking at
        one radius (or a handful) at a time.

        A physical radius band this thin (tol, meters) still contains a
        huge number of raw surfels - hundreds of thousands, easily, on a
        real .snc surface - spread over a genuine spread of x/c AND
        Cf (band width, real turbulence, not just noise). Returning every
        one of them makes a "curve" that's really a dense cloud once
        plotted. n_chord_bins bins x/c into that many equal bins and
        averages Cf within each one, which is what actually produces a
        readable curve - see plot_cf_radii().

        Parameters
        ----------
        radii : array-like of float
            Target physical radii [m] (from the rotation axis - see
            _radius()).
        tol : float
            Half-width [m] of the radius band selected around each target.
        n_chord_bins : int or None
            Number of x/c bins to average Cf within. None returns every
            raw surfel instead (unsorted-looking cloud) - useful mainly
            for inspecting the raw scatter this averaging is smoothing
            over.
        span_min, span_max : float, optional
            Keep only points with span_min <= span <= span_max (centered
            Cartesian span - see _span_chord()), BEFORE computing this
            band's chord min/max. REQUIRED in practice on any file with
            more than one blade sharing the same radius range (e.g. one
            face/file covering the whole rotor, as this project's own
            .snc files do): without it, a radius band picks up every
            blade's surfels, and x/c gets normalized against their
            COMBINED chord range - confirmed to produce a spurious extra
            peak at both x/c=0 AND x/c=1 (Cf's real near-edge rise gets
            duplicated once per blade instead of appearing once), which
            is exactly what friction_lines() already guards against with
            its own span_min/span_max (see that method's docstring) - this
            parameter was missing here even though the same underlying
            data has the same two-blade problem; not caught until a real
            plot (Cf vs. x/c at several radii) showed the extra peak. No
            default here either: there's no reliable automatic value, it
            depends on this case's own span layout.
        reverse_chord : bool
            If True, flip which raw chord extreme maps to x/c=0 vs. x/c=1.
            The raw chord axis has no inherent leading/trailing-edge
            orientation - x/c=0 was arbitrarily assigned to the minimum
            chord value. Off by default, which put Cf's real near-LE peak
            at x/c=1 instead of x/c=0 on this project's own case
            (confirmed against images/cf/skin-friction_radii_plot.png, a
            trusted reference plot with the peak at x/c=0) - the same
            orientation ambiguity already documented for Cp
            (SurfaceVariable.at_radii()'s reverse_chord). Check which
            orientation is right per case: Cf should peak sharply near the
            leading edge (thin boundary layer, high wall shear) and decay
            toward the trailing edge - if that peak shows up at x/c=1
            instead of x/c=0, set this to True.

        Returns
        -------
        dict[float, tuple[np.ndarray, np.ndarray]]
            radius -> (xc, cf), both sorted by xc. xc is bin-center
            positions if n_chord_bins is set, else one entry per raw
            surfel.
        '''

        r = self._radius(surface)
        span, chord = self._span_chord(surface)
        cf = self.cf(surface=surface, frame=frame, component=component)

        span_mask = np.ones(len(span), dtype=bool)
        if span_min is not None:
            span_mask &= span >= span_min
        if span_max is not None:
            span_mask &= span <= span_max

        curves = {}

        for r_target in radii:

            mask = span_mask & (np.abs(r - r_target) < tol)
            n_sel = int(mask.sum())

            if n_sel < 10:
                raise ValueError(
                    f"Only {n_sel} points within {tol} m of r={r_target} (after span_min/span_max "
                    "cropping) - widen tol, check r_target, or check span_min/span_max."
                )

            c = chord[mask]
            c_min, c_max = c.min(), c.max()
            xc = (c - c_min) / (c_max - c_min)
            if reverse_chord:
                xc = 1 - xc
            cf_sel = cf[mask]

            if n_chord_bins is None:
                order = np.argsort(xc)
                curves[r_target] = (xc[order], cf_sel[order])
                continue

            bin_edges = np.linspace(0, 1, n_chord_bins + 1)
            bin_idx = np.clip(np.digitize(xc, bin_edges) - 1, 0, n_chord_bins - 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            cf_mean = np.full(n_chord_bins, np.nan)
            counts = np.bincount(bin_idx, minlength=n_chord_bins)
            sums = np.bincount(bin_idx, weights=cf_sel, minlength=n_chord_bins)
            has_data = counts > 0
            cf_mean[has_data] = sums[has_data] / counts[has_data]

            curves[r_target] = (bin_centers[has_data], cf_mean[has_data])

        return curves

    def plot_cf_radii(self, radii, surface: str = 'Upper', frame: int = None, component: str = None,
                       tol: float = 0.0015, n_chord_bins: int = 150, span_min: float = None,
                       span_max: float = None, reverse_chord: bool = False, cmap: str = 'cividis',
                       ax=None, savepath: str = None, dpi: int = 150):

        '''
        Plot Cf vs. local chordwise position for several radii on one set
        of axes. See cf_at_radii() for what the two axes mean, its caveat
        on the chordwise coordinate, what n_chord_bins does, why
        span_min/span_max matter (isolating one blade on a multi-blade
        file - required in practice, see cf_at_radii()'s docstring for
        the two-blade artifact this avoids), and why reverse_chord matters
        (x/c has no inherent LE/TE orientation - check per case; Cf should
        peak near the leading edge, x/c=0 by default here).

        Plotted as a line by default (n_chord_bins set): cf_at_radii()'s
        per-bin averaging already turns the raw, noisy surfel cloud into a
        clean, properly x/c-ordered curve, so connecting points is
        legitimate. Pass n_chord_bins=None to see the raw, unaveraged
        surfel scatter instead (plotted as a scatter, not a line - with
        that much raw noise, connecting points would draw a dense zigzag
        that reads as a filled, illegible cluster rather than a curve).

        A horizontal zero line is added whenever component is not None,
        since the point of a signed component is spotting where curves
        cross it. Default colormap cividis (not viridis) and a grid, to
        match this project's other radius-colored plots (SurfaceVariable's
        plot_at_radii()/plot_cp_radii()).

        Returns
        -------
        (fig, ax)
        '''

        curves = self.cf_at_radii(radii, surface=surface, frame=frame, component=component, tol=tol,
                                   n_chord_bins=n_chord_bins, span_min=span_min, span_max=span_max,
                                   reverse_chord=reverse_chord)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(curves)))

        for color, (r_target, (xc, cf_curve)) in zip(colors, curves.items()):
            if n_chord_bins is None:
                ax.scatter(xc, cf_curve, s=4, alpha=0.5, color=color, label=f'r = {r_target:g} m')
            else:
                ax.plot(xc, cf_curve, color=color, label=f'r = {r_target:g} m')

        if component is not None:
            ax.axhline(0, color='k', linewidth=0.8, zorder=0)

        case_label = 'average' if frame is None else f'instantaneous, frame {frame}'
        cf_label = 'Cf' if component is None else f'Cf ({component})'

        ax.set_xlabel('x/c (local, per-radius - see cf_at_radii)')
        ax.set_ylabel(cf_label)
        ax.set_title(f'{surface} surface - {cf_label} - {case_label}')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax

    def friction_lines(self, surface=('Upper', 'Lower'), frame: int = None, span_min: float = None,
                        span_max: float = None, cf_clip_percentile: float = 99, n_arrows: int = 2000,
                        marker_size: float = 1, cmap: str = 'viridis', cbar_label: str = 'cf',
                        figsize: tuple = None, savepath: str = None, dpi: int = 150):

        '''
        Friction lines: a dense scatter of Cf magnitude over the blade
        surface (raw Cartesian span/chord - see class docstring), with a
        sparse quiver of the wall-shear DIRECTION on top - one row per
        surface (Upper/Lower by default, stacked), matching
        friction_lines_normal_split.png from earlier this project.

        Deliberately NOT interpolated onto a grid first (an earlier
        version of this method did, via griddata/streamplot, and was both
        slow - rebuilding a Delaunay triangulation over >10M points - and
        introduced visible interpolation artifacts). A real .snc surface
        has tens of millions of surfels; plotting them directly as small
        points already reads as a continuous field.

        Parameters
        ----------
        surface : str or tuple of str
            'Upper', 'Lower', or a tuple of both (default) for a stacked
            two-row figure sharing the span (x) axis.
        span_min, span_max : float, optional
            Keep only points with span_min <= span <= span_max (centered
            Cartesian span - see _span_chord()). Use this to cut away the
            hub/root region, e.g. span_min=0.03 isolates one blade half on
            a two-bladed rotor centered at span=0 - there's no reliable
            automatic default (an earlier radius-fraction heuristic here
            was wrong for this project's own data), so this is left
            unset (no cropping) unless you pass it.
        cf_clip_percentile : float
            Colors are clipped to [0, this percentile of Cf magnitude]
            before plotting, since without clipping a handful of extreme
            surfels (e.g. right at a sharp edge) wash out the entire color
            scale. Set to 100 to disable.
        n_arrows : int
            Number of surfels randomly sampled for the direction quiver -
            one arrow per surfel would be illegible.
        marker_size : float
            Scatter marker size (matplotlib's `s`, points^2).

        Returns
        -------
        (fig, axes)
            axes is a single Axes if surface is a single string, otherwise
            a list of Axes in the same order as surface.
        '''

        if self.rho_ref is None or self.rpm is None:
            raise ValueError("rho_ref and rpm must be set (in __init__) to compute Cf.")

        surfaces = (surface,) if isinstance(surface, str) else tuple(surface)
        case_label = 'average' if frame is None else f'instantaneous, frame {frame}'
        rng = np.random.default_rng(0)

        if figsize is None:
            figsize = (14, 4.5 * len(surfaces))

        fig, axes = plt.subplots(len(surfaces), 1, figsize=figsize, sharex=True, squeeze=False)
        axes = axes[:, 0]

        for ax, surf in zip(axes, surfaces):

            span, chord = self._span_chord(surf)
            cf_mag = self.cf(surface=surf, frame=frame, component=None)
            tau_chord = self.cf(surface=surf, frame=frame, component='chordwise')
            tau_span = self.cf(surface=surf, frame=frame, component='spanwise')

            mask = np.ones(len(span), dtype=bool)
            if span_min is not None:
                mask &= span >= span_min
            if span_max is not None:
                mask &= span <= span_max

            span_sel, chord_sel = span[mask], chord[mask]
            cf_sel, tau_chord_sel, tau_span_sel = cf_mag[mask], tau_chord[mask], tau_span[mask]

            vmax = np.percentile(cf_sel, cf_clip_percentile)
            cf_clipped = np.clip(cf_sel, 0, vmax)

            sc = ax.scatter(span_sel, chord_sel, c=cf_clipped, s=marker_size, cmap=cmap, vmin=0, vmax=vmax)
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(cbar_label)

            idx = rng.choice(len(span_sel), size=min(n_arrows, len(span_sel)), replace=False)
            mag = np.hypot(tau_span_sel[idx], tau_chord_sel[idx])
            mag[mag == 0] = 1
            ax.quiver(span_sel[idx], chord_sel[idx], tau_span_sel[idx] / mag, tau_chord_sel[idx] / mag,
                      color='white', scale=60, width=0.002, alpha=0.8)

            ax.set_ylabel('chord [m]')
            ax.set_aspect('equal')
            ax.set_title(f'{surf} ({case_label})')

        axes[-1].set_xlabel('span [m]')
        fig.suptitle('Friction lines')
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, (axes[0] if len(axes) == 1 else list(axes))
