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
                    'normals': np.column_stack([geo['Normal_X'][:], geo['Normal_Y'][:], geo['Normal_Z'][:]]),
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

        cf_label = {None: r'$C_f$', 'chordwise': r'$C_{f,c}$', 'spanwise': r'$C_{f,s}$'}[component]

        ax.set_xlabel(r'$x/c$ [-]')
        ax.set_ylabel(f'{cf_label} [-]')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax

    def separation_line(self, surface: str = 'Upper', frame: int = None, span_min: float = None,
                         span_max: float = None, n_span_bins: int = 200, n_chord_bins: int = 200,
                         chord_percentile: float = 0.1, reverse_chord: bool = False):

        '''
        Separation/reattachment line: every span location where chordwise
        Cf (tau[chord_axis] / q_ref) crosses zero - the near-wall flow
        reversal signature already noted in the class docstring's "Cf
        normalization" and README.md's "Equations" section (magnitude
        can only dip toward zero; the signed chordwise component actually
        crosses it, which is the point of having it).

        Restricted to ONE blade section (span_min/span_max) for the same
        reason as everywhere else in this class (cf_at_radii()/
        friction_lines()): a multi-blade file's raw span axis mixes chord
        ranges across blades, which would corrupt the local x/c each
        crossing's chord position is measured against.

        How it works: the selected span range is partitioned into
        n_span_bins equal-width bins (same idea as cf_at_radii()'s radius
        bands, but a continuous sweep across span instead of a handful of
        fixed target radii). Within each span bin, chordwise Cf is
        averaged into n_chord_bins x/c bins using the SAME per-band,
        percentile-normalized x/c convention as cf_at_radii() (see that
        method's docstring for chord_percentile/reverse_chord - pass the
        SAME reverse_chord you used there/in plot_cf_radii(), otherwise
        the two disagree on which end is the leading edge). Adjacent
        chord bins with a sign change are linearly interpolated to
        localize each crossing. A span bin can produce zero, one, or
        several crossings (e.g. a separation bubble followed by another
        further aft) - all are kept, not just the first.

        Labeling ('separation' vs. 'reattachment'): NOT based on the raw
        sign of Cf (+ -> - vs. - -> +). Which physical direction is
        "positive" chordwise Cf depends on how this mesh's chord_axis
        happens to be oriented in the .snc file - an arbitrary modeling
        choice, not a physical convention - so a fixed
        "+ to - is separation" rule is only correct for one of the two
        possible axis orientations, and silently wrong (reattachment
        appearing to precede its own separation, which is not physically
        possible) for the other. Confirmed on this project's own case:
        exactly this inversion showed up in
        friction_lines_test_separation_overlay.png before this fix.
        Instead, crossings within each span bin are labeled purely by
        ORDER along x/c: the 1st, 3rd, 5th, ... crossing (0-indexed: 0,
        2, 4, ...) is 'separation' (entering a reversed-flow region), the
        2nd, 4th, 6th, ... is 'reattachment' (leaving it) - always
        alternating, always paired, regardless of which raw sign happens
        to mean "attached" on this mesh. A bin with an odd number of
        crossings has one unpaired 'separation' left over: the reversed
        region it opened extends past the last chord bin with data
        (e.g. all the way to the trailing edge) rather than closing
        within the resolved x/c range - not a bug, a real open region.
        Each pair shares the same 'pair_id' (see Returns), so
        plot_separation_line() can draw the bubble extent between them.

        Span bins with fewer than 10 points (after span_min/span_max
        cropping), or whose chord-bin curve has too few populated bins to
        localize a crossing, are silently skipped - this can break an
        otherwise-continuous line into gaps rather than interpolating
        across them; increase n_span_bins/reduce n_chord_bins if that
        happens somewhere important.

        Parameters
        ----------
        n_span_bins : int
            Number of span bins the (cropped) span range is partitioned
            into - the resolution of the returned line along the span
            axis. Higher than cf_at_radii()'s handful of fixed radii by
            design - this is meant to trace a continuous line, not spot-
            check a few stations.
        n_chord_bins : int
            Number of x/c bins Cf is averaged into within each span bin -
            same purpose as cf_at_radii()'s parameter of the same name
            (turns a noisy raw-surfel cloud into a smooth curve to find
            crossings on); higher resolves a sharper crossing location at
            the cost of noisier bins.

        Returns
        -------
        list[dict]
            One entry per crossing found, sorted by span, each with keys
            'span' [m], 'chord' [m] (both raw Cartesian, centered - see
            _span_chord(), directly usable as (x, y) on friction_lines()'s
            axes), 'r' [m] (mean physical radius of the span bin the
            crossing came from), 'xc' (local x/c position, 0-1), 'kind'
            ('separation' or 'reattachment', by order - see above), and
            'pair_id' (int, shared by a matched separation/reattachment
            pair from the same span bin - -1 for an unpaired trailing
            crossing).
        '''

        if self.rho_ref is None or self.rpm is None:
            raise ValueError("rho_ref and rpm must be set (in __init__) to compute Cf.")

        span, chord = self._span_chord(surface)
        r = self._radius(surface)
        tau_chord = self.cf(surface=surface, frame=frame, component='chordwise')

        mask = np.ones(len(span), dtype=bool)
        if span_min is not None:
            mask &= span >= span_min
        if span_max is not None:
            mask &= span <= span_max

        span_m, chord_m, r_m, cf_m = span[mask], chord[mask], r[mask], tau_chord[mask]

        if len(span_m) < 10:
            raise ValueError(
                f"Only {len(span_m)} points after span_min/span_max cropping - "
                "check span_min/span_max."
            )

        span_edges = np.linspace(span_m.min(), span_m.max(), n_span_bins + 1)
        chord_edges = np.linspace(0, 1, n_chord_bins + 1)
        chord_centers = (chord_edges[:-1] + chord_edges[1:]) / 2

        points = []
        next_pair_id = 0

        for i in range(n_span_bins):

            in_bin = (span_m >= span_edges[i]) & (
                span_m < span_edges[i + 1] if i < n_span_bins - 1 else span_m <= span_edges[i + 1]
            )
            if in_bin.sum() < 10:
                continue

            c = chord_m[in_bin]
            cf_sel = cf_m[in_bin]
            span_center = 0.5 * (span_edges[i] + span_edges[i + 1])
            r_mean = r_m[in_bin].mean()

            c_min, c_max = np.percentile(c, [chord_percentile, 100 - chord_percentile])
            valid = (c >= c_min) & (c <= c_max)
            if valid.sum() < 10:
                continue

            xc_norm = (c[valid] - c_min) / (c_max - c_min)
            xc = 1 - xc_norm if reverse_chord else xc_norm
            cf_valid = cf_sel[valid]

            bin_idx = np.clip(np.digitize(xc, chord_edges) - 1, 0, n_chord_bins - 1)
            counts = np.bincount(bin_idx, minlength=n_chord_bins)
            sums = np.bincount(bin_idx, weights=cf_valid, minlength=n_chord_bins)
            has_data = counts > 0

            cf_curve = np.full(n_chord_bins, np.nan)
            cf_curve[has_data] = sums[has_data] / counts[has_data]

            xc_curve = chord_centers[has_data]
            cf_curve = cf_curve[has_data]
            order = np.argsort(xc_curve)
            xc_curve, cf_curve = xc_curve[order], cf_curve[order]

            bin_crossings = []

            for j in range(len(cf_curve) - 1):

                f0, f1 = cf_curve[j], cf_curve[j + 1]
                if f0 == 0 or f1 == 0 or f0 * f1 > 0:
                    continue

                t = f0 / (f0 - f1)
                xc_cross = xc_curve[j] + t * (xc_curve[j + 1] - xc_curve[j])
                xc_norm_cross = 1 - xc_cross if reverse_chord else xc_cross
                chord_cross = c_min + xc_norm_cross * (c_max - c_min)

                bin_crossings.append({'span': span_center, 'chord': chord_cross, 'r': r_mean, 'xc': xc_cross})

            for k, p in enumerate(bin_crossings):
                p['kind'] = 'separation' if k % 2 == 0 else 'reattachment'
                if k % 2 == 0:
                    p['pair_id'] = next_pair_id if (k + 1) < len(bin_crossings) else -1
                else:
                    p['pair_id'] = bin_crossings[k - 1]['pair_id']
                    next_pair_id += 1

            points.extend(bin_crossings)

        return sorted(points, key=lambda p: p['span'])

    def plot_separation_line(self, ax, points, colors=('red', 'cyan'), marker_size: float = 45,
                              connect_pairs: bool = False, line_color: str = 'black',
                              linewidth: float = 1.8, label: bool = True):

        '''
        Overlay separation_line()'s crossings on an existing Axes plotted
        in (span, chord) coordinates - e.g. one of friction_lines()'s
        per-surface axes (or pass show_separation_line=True to
        friction_lines() instead of calling this directly).

        Sized/colored to stay visible against friction_lines()'s dense
        background scatter (large markers, black edge, high-contrast
        colors) - the defaults here are deliberately bigger than a
        typical matplotlib scatter default for that reason.

        Parameters
        ----------
        ax : matplotlib Axes
        points : list[dict]
            separation_line()'s return value.
        colors : (str, str)
            Marker colors for ('separation', 'reattachment') points.
        connect_pairs : bool
            If True, draw a short line between each matched separation/
            reattachment pair (same 'pair_id' - see separation_line()),
            marking the reversed-flow region's chordwise extent at that
            span station. Off by default - on a dense case (many span
            bins) the connecting lines packed side by side read as a
            solid black wall rather than individually legible segments;
            the markers alone already show both endpoints. Unpaired
            crossings (pair_id == -1) are never connected either way.
        label : bool
            Add a legend entry for each kind present (skipped if ax
            already has a legend you're managing yourself - call
            ax.legend() again after this to pick up the new entries).
        '''

        if connect_pairs:
            by_pair = {}
            for p in points:
                if p['pair_id'] == -1:
                    continue
                by_pair.setdefault(p['pair_id'], []).append(p)
            for pair in by_pair.values():
                if len(pair) != 2:
                    continue
                a, b = pair
                ax.plot([a['span'], b['span']], [a['chord'], b['chord']], color=line_color,
                        linewidth=linewidth, alpha=0.85, zorder=4)

        for kind, color in zip(('separation', 'reattachment'), colors):
            sel = [p for p in points if p['kind'] == kind]
            if not sel:
                continue
            ax.scatter([p['span'] for p in sel], [p['chord'] for p in sel], s=marker_size,
                       color=color, label=kind if label else None, zorder=5,
                       edgecolors='k', linewidths=0.6)

    def save_separation_line(self, points, filepath: str):

        '''
        Write separation_line()'s crossings to a plain-text file, one row
        per crossing: span, chord (both physical, meters, raw Cartesian -
        see separation_line()'s Returns), r, xc, kind, pair_id -
        tab-separated with a header line.
        '''

        with open(filepath, 'w') as f:
            f.write('span_m\tchord_m\tr_m\txc\tkind\tpair_id\n')
            for p in points:
                f.write(f"{p['span']:.6f}\t{p['chord']:.6f}\t{p['r']:.6f}\t{p['xc']:.4f}\t"
                        f"{p['kind']}\t{p['pair_id']}\n")

    def migration_line(self, surface: str = 'Upper', frame: int = None, span_min: float = None,
                        span_max: float = None, n_span_bins: int = 200, n_chord_bins: int = 200,
                        chord_percentile: float = 0.1, reverse_chord: bool = False,
                        flip_direction: bool = False, edge_crop: float = 0.05):

        '''
        Spanwise migration-reversal line: every location where SPANWISE
        Cf (tau[span_axis] / q_ref) crosses zero - where near-wall flow
        switches between migrating toward the tip ("outward", e.g.
        classic centrifugal pumping in a rotating boundary layer) and
        migrating toward the root ("inward") - this is the sign flip
        directly visible in this project's own
        friction_lines_test_cf_spanwise_avg.png (inner radii mostly
        positive/outward, an outer-radius station dipping negative/
        inward over part of its chord).

        NOT a separation/reattachment line - that's the CHORDWISE
        component's zero crossing (see separation_line()), a physically
        different phenomenon (chordwise flow reversal vs. spanwise
        migration direction). Kept as a separate method rather than a
        mode flag on separation_line(), to avoid overloading one API
        with two different physical meanings.

        Same binning architecture as separation_line() (same span_min/
        span_max blade-isolation requirement, same per-span-band,
        percentile-normalized x/c convention via chord_percentile/
        reverse_chord - pass the SAME reverse_chord you use with
        plot_cf_radii()/separation_line() on this case, or the two
        disagree on which end is the leading edge) - only the Cf
        component being scanned for zero crossings changes.

        Labeling ('outward' vs. 'inward'): UNLIKE separation_line() -
        which can only label crossings by ORDER, since chordwise Cf's
        sign has no fixed physical meaning (it depends on this mesh's
        arbitrary chord_axis orientation, discovered the hard way via
        the reverse_chord fix) - spanwise Cf's sign CAN be given a real
        physical meaning here: span is used as-is, the raw absolute
        Cartesian coordinate (never renormalized per band the way x/c
        is), and span_min/span_max is already how a single blade is
        isolated, running from the hub outward to the tip - so on that
        selected half, increasing span consistently means "toward the
        tip". A crossing from positive to negative spanwise Cf is
        therefore labeled 'inward' (flow that was migrating toward the
        tip starts migrating toward the root); negative to positive is
        'outward'. This assumption was validated on this project's own
        case (span runs from ~0.02 to ~0.125 m = r_tip on the
        span_min=0.02-cropped half, confirming span does increase toward
        the tip there) - if a different case's span_axis happens to
        point the opposite way on whichever half you cropped, set
        flip_direction=True to swap the meaning; there's no way to detect
        this automatically, check per case the same way reverse_chord is
        checked.

        **Noise near the leading edge**: confirmed on this project's own
        case - right at the LE, near-wall flow is nearly pure chordwise
        (splitting into the pressure/suction sides), so spanwise Cf sits
        very close to zero AND has a genuinely large local gradient there
        (not just noise - the curve swings hard over a very short x/c
        distance right at the LE), which can flip sign from one chord bin
        to the next for no meaningful physical reason, producing a dense
        band of rapidly-alternating 'outward'/'inward' crossings that
        trace an LE artifact, not a real migration boundary.

        edge_crop drops crossings within edge_crop of x/c=0 or x/c=1 (the
        SAME parameter/idea as SurfaceVariable.at_radii()'s edge_crop -
        see that method's docstring). This went through two failed
        attempts before landing here, worth recording: an amplitude-based
        filter (reject a crossing unless both bracketing bins' |Cf|
        exceeds some fraction of a reference scale) seemed like the
        obvious fix, but broke twice - once using each span bin's own
        curve max as the reference (a large LE excursion in one bin
        could inflate its own threshold enough to asymmetrically kill the
        SMALLER-magnitude half of an otherwise genuine pair elsewhere in
        that same bin, silently producing one-sided output - confirmed:
        68 crossings, ALL labeled 'inward', zero 'outward'), and again
        using one global scale across the whole case (the LE's own large
        gradient dominated that global scale too, so the filter ended up
        keeping only the single largest - i.e. LE-adjacent - crossing per
        bin and discarding the genuine mid-chord/tip-region signal
        entirely - confirmed: every surviving crossing sat at
        xc < 0.007). Excluding the LE by POSITION rather than by
        AMPLITUDE sidesteps both failure modes, since it doesn't matter
        how large or small the LE's own Cf swing is - it's just never
        considered. Set edge_crop=0 to see every raw crossing (including
        the LE artifact) if you want to inspect it yourself instead.

        Parameters
        ----------
        (same as separation_line() - n_span_bins, n_chord_bins,
        chord_percentile, reverse_chord - plus)
        flip_direction : bool
            Swap the 'outward'/'inward' meaning - see above.
        edge_crop : float
            Drop crossings within this fraction of x/c=0 or x/c=1 - see
            "Noise near the leading edge" above.

        Returns
        -------
        list[dict]
            Same shape as separation_line(): 'span' [m], 'chord' [m],
            'r' [m], 'xc', 'kind' ('outward' or 'inward'), 'pair_id'
            (crossings from the same span bin that bracket one
            migration-reversed patch share an id; -1 if unpaired - see
            separation_line()'s Returns for what that means).
        '''

        if self.rho_ref is None or self.rpm is None:
            raise ValueError("rho_ref and rpm must be set (in __init__) to compute Cf.")

        span, chord = self._span_chord(surface)
        r = self._radius(surface)
        tau_span = self.cf(surface=surface, frame=frame, component='spanwise')

        mask = np.ones(len(span), dtype=bool)
        if span_min is not None:
            mask &= span >= span_min
        if span_max is not None:
            mask &= span <= span_max

        span_m, chord_m, r_m, cf_m = span[mask], chord[mask], r[mask], tau_span[mask]

        if len(span_m) < 10:
            raise ValueError(
                f"Only {len(span_m)} points after span_min/span_max cropping - "
                "check span_min/span_max."
            )

        span_edges = np.linspace(span_m.min(), span_m.max(), n_span_bins + 1)
        chord_edges = np.linspace(0, 1, n_chord_bins + 1)
        chord_centers = (chord_edges[:-1] + chord_edges[1:]) / 2

        points = []
        next_pair_id = 0

        for i in range(n_span_bins):

            in_bin = (span_m >= span_edges[i]) & (
                span_m < span_edges[i + 1] if i < n_span_bins - 1 else span_m <= span_edges[i + 1]
            )
            if in_bin.sum() < 10:
                continue

            c = chord_m[in_bin]
            cf_sel = cf_m[in_bin]
            span_center = 0.5 * (span_edges[i] + span_edges[i + 1])
            r_mean = r_m[in_bin].mean()

            c_min, c_max = np.percentile(c, [chord_percentile, 100 - chord_percentile])
            valid = (c >= c_min) & (c <= c_max)
            if valid.sum() < 10:
                continue

            xc_norm = (c[valid] - c_min) / (c_max - c_min)
            xc = 1 - xc_norm if reverse_chord else xc_norm
            cf_valid = cf_sel[valid]

            bin_idx = np.clip(np.digitize(xc, chord_edges) - 1, 0, n_chord_bins - 1)
            counts = np.bincount(bin_idx, minlength=n_chord_bins)
            sums = np.bincount(bin_idx, weights=cf_valid, minlength=n_chord_bins)
            has_data = counts > 0

            cf_curve = np.full(n_chord_bins, np.nan)
            cf_curve[has_data] = sums[has_data] / counts[has_data]

            xc_curve = chord_centers[has_data]
            cf_curve = cf_curve[has_data]
            order = np.argsort(xc_curve)
            xc_curve, cf_curve = xc_curve[order], cf_curve[order]

            bin_crossings = []

            for j in range(len(cf_curve) - 1):

                f0, f1 = cf_curve[j], cf_curve[j + 1]
                if f0 == 0 or f1 == 0 or f0 * f1 > 0:
                    continue

                t = f0 / (f0 - f1)
                xc_cross = xc_curve[j] + t * (xc_curve[j + 1] - xc_curve[j])
                if xc_cross < edge_crop or xc_cross > 1 - edge_crop:
                    continue
                xc_norm_cross = 1 - xc_cross if reverse_chord else xc_cross
                chord_cross = c_min + xc_norm_cross * (c_max - c_min)

                kind = 'inward' if f0 > 0 else 'outward'
                if flip_direction:
                    kind = 'outward' if kind == 'inward' else 'inward'

                bin_crossings.append({
                    'span': span_center, 'chord': chord_cross, 'r': r_mean,
                    'xc': xc_cross, 'kind': kind,
                })

            for k, p in enumerate(bin_crossings):
                if k % 2 == 0:
                    p['pair_id'] = next_pair_id if (k + 1) < len(bin_crossings) else -1
                else:
                    p['pair_id'] = bin_crossings[k - 1]['pair_id']
                    next_pair_id += 1

            points.extend(bin_crossings)

        return sorted(points, key=lambda p: p['span'])

    def plot_migration_line(self, ax, points, colors=('orange', 'purple'), marker_size: float = 45,
                             connect_pairs: bool = False, line_color: str = 'black',
                             linewidth: float = 1.8, label: bool = True):

        '''
        Overlay migration_line()'s crossings - same visual convention as
        plot_separation_line() (large, black-edged, high-contrast
        markers; connect_pairs off by default for the same reason), but
        triangular markers (vs. plot_separation_line()'s circles) so the
        two overlays stay visually distinct if shown together.

        Parameters
        ----------
        ax : matplotlib Axes
        points : list[dict]
            migration_line()'s return value.
        colors : (str, str)
            Marker colors for ('outward', 'inward') points.
        connect_pairs : bool
            See plot_separation_line() - same meaning, same off-by-
            default reasoning.
        label : bool
            Add a legend entry for each kind present.
        '''

        if connect_pairs:
            by_pair = {}
            for p in points:
                if p['pair_id'] == -1:
                    continue
                by_pair.setdefault(p['pair_id'], []).append(p)
            for pair in by_pair.values():
                if len(pair) != 2:
                    continue
                a, b = pair
                ax.plot([a['span'], b['span']], [a['chord'], b['chord']], color=line_color,
                        linewidth=linewidth, alpha=0.85, zorder=4)

        for kind, color in zip(('outward', 'inward'), colors):
            sel = [p for p in points if p['kind'] == kind]
            if not sel:
                continue
            ax.scatter([p['span'] for p in sel], [p['chord'] for p in sel], s=marker_size,
                       color=color, marker='^', label=kind if label else None, zorder=5,
                       edgecolors='k', linewidths=0.6)

    def save_migration_line(self, points, filepath: str):

        '''
        Write migration_line()'s crossings to a plain-text file - same
        format as save_separation_line(): span_m, chord_m, r_m, xc,
        kind, pair_id, tab-separated with a header line.
        '''

        with open(filepath, 'w') as f:
            f.write('span_m\tchord_m\tr_m\txc\tkind\tpair_id\n')
            for p in points:
                f.write(f"{p['span']:.6f}\t{p['chord']:.6f}\t{p['r']:.6f}\t{p['xc']:.4f}\t"
                        f"{p['kind']}\t{p['pair_id']}\n")

    def critical_points(self, surface: str = 'Upper', frame: int = None, span_min: float = None,
                         span_max: float = None, n_span_bins: int = 150, n_chord_bins: int = 150,
                         chord_percentile: float = 0.5, magnitude_percentile: float = 2.0,
                         min_count: int = 5):

        '''
        Skin-friction-line critical points: locations where the ENTIRE
        wall-shear vector (chordwise Cf, spanwise Cf) vanishes
        simultaneously - not just one component's zero crossing (see
        separation_line()/migration_line(), which each scan a single
        component along a single direction). Per Lighthill's theorem
        (see Tobak & Peake, 1982, "Topology of Three-Dimensional
        Separated Flows" - the standard reference for this kind of
        surface-flow topology analysis), any point a real 3D flow's
        surface streamlines converge to, diverge from, or spiral around
        MUST be a point of zero skin friction - so genuine 3D flow
        structures, including vortex footprints specifically, can be
        located this way.

        Unlike separation_line()/migration_line() (which bin per span
        band using a LOCALLY renormalized x/c, since each band's own
        chord min/max differs), this bins the raw surfel cloud onto a
        genuine 2D grid in ABSOLUTE physical (span, chord) coordinates
        [m] - a real 2D neighborhood search needs a consistent
        coordinate system across neighboring cells, which a per-band-
        renormalized x/c doesn't give. One consequence: no
        reverse_chord parameter here - raw physical chord already has a
        real, consistent geometric meaning on its own, unlike x/c.

        How it works:
          1. The selected region (span_min/span_max-cropped, same
             blade-isolation requirement as everywhere else in this
             class) is binned onto an n_span_bins x n_chord_bins grid;
             each cell holds the mean chordwise Cf, spanwise Cf, and Cf
             magnitude of the surfels falling in it. Cells with fewer
             than min_count surfels are treated as unreliable (excluded
             from candidacy and from being used as a neighbor).
          2. Candidate critical points are interior cells (a full 3x3
             neighborhood, needed for step 3) whose Cf magnitude is in
             the lowest magnitude_percentile% of the whole grid AND a
             local minimum among their (reliable) neighbors - guards
             against picking out many cells along the edge of one broad
             low-Cf region instead of its actual center.
          3. Each candidate is classified via the local Jacobian of
             (chordwise Cf, spanwise Cf) with respect to (chord, span),
             estimated by central finite differences on the grid, using
             its eigenvalues:
                 'node'   : real eigenvalues, same sign - surface lines
                            all converge to (or diverge from) this point
                            (a 3D separation/attachment node)
                 'saddle' : real eigenvalues, opposite sign - lines pass
                            through/around it without converging (a
                            typical reattachment saddle)
                 'focus'  : complex eigenvalues - lines spiral around it
                            - the actual footprint of a VORTEX core
                            (leading-edge vortex, corner/horseshoe
                            vortex, ...), not just an ordinary
                            separation/reattachment feature

        This is a real but approximate, resolution- and noise-sensitive
        tool - a coarse grid can merge nearby critical points or miss
        weak ones; a genuinely noisy Cf field can manufacture a
        spurious low-magnitude cell that isn't a real critical point.
        Treat results as candidates to inspect against friction_lines()'s
        own quiver pattern, not as ground truth by themselves.

        Parameters
        ----------
        n_span_bins, n_chord_bins : int
            Grid resolution - higher resolves closely-spaced critical
            points better, at the cost of noisier per-cell averages
            (fewer surfels per cell) and a less reliable finite-
            difference Jacobian.
        chord_percentile : float
            The grid's chord extent is the [chord_percentile,
            100-chord_percentile] percentile of chord in the cropped
            selection, not literal min/max - same outlier-guarding
            purpose as elsewhere in this class.
        magnitude_percentile : float
            Candidates must have Cf magnitude within this bottom
            percentile of the whole (reliable-cell) grid, in addition to
            being a local minimum - avoids treating an ordinary
            "somewhat low but not really zero" cell as a critical point
            just because it's a local minimum relative to its immediate
            neighbors.
        min_count : int
            Minimum surfels a grid cell needs to be treated as reliable.

        Returns
        -------
        list[dict]
            One entry per critical point found, each with 'span' [m],
            'chord' [m] (raw Cartesian, centered - see _span_chord()),
            'r' [m], 'cf_mag' (the cell's Cf magnitude), and 'kind'
            ('node', 'saddle', or 'focus').
        '''

        if self.rho_ref is None or self.rpm is None:
            raise ValueError("rho_ref and rpm must be set (in __init__) to compute Cf.")

        span, chord = self._span_chord(surface)
        r = self._radius(surface)
        cf_mag = self.cf(surface=surface, frame=frame, component=None)
        tau_chord = self.cf(surface=surface, frame=frame, component='chordwise')
        tau_span = self.cf(surface=surface, frame=frame, component='spanwise')

        mask = np.ones(len(span), dtype=bool)
        if span_min is not None:
            mask &= span >= span_min
        if span_max is not None:
            mask &= span <= span_max

        span_m, chord_m, r_m = span[mask], chord[mask], r[mask]
        mag_m, u_m, v_m = cf_mag[mask], tau_chord[mask], tau_span[mask]

        if len(span_m) < 10:
            raise ValueError(
                f"Only {len(span_m)} points after span_min/span_max cropping - "
                "check span_min/span_max."
            )

        span_lo, span_hi = span_m.min(), span_m.max()
        chord_lo, chord_hi = np.percentile(chord_m, [chord_percentile, 100 - chord_percentile])

        span_edges = np.linspace(span_lo, span_hi, n_span_bins + 1)
        chord_edges = np.linspace(chord_lo, chord_hi, n_chord_bins + 1)
        span_centers = (span_edges[:-1] + span_edges[1:]) / 2
        chord_centers = (chord_edges[:-1] + chord_edges[1:]) / 2

        in_grid = (chord_m >= chord_lo) & (chord_m <= chord_hi)
        span_idx = np.clip(np.digitize(span_m[in_grid], span_edges) - 1, 0, n_span_bins - 1)
        chord_idx = np.clip(np.digitize(chord_m[in_grid], chord_edges) - 1, 0, n_chord_bins - 1)
        flat_idx = span_idx * n_chord_bins + chord_idx
        n_cells = n_span_bins * n_chord_bins

        def grid_mean(values_full):
            vals = values_full[in_grid]
            sums = np.bincount(flat_idx, weights=vals, minlength=n_cells)
            counts = np.bincount(flat_idx, minlength=n_cells)
            out = np.full(n_cells, np.nan)
            has = counts > 0
            out[has] = sums[has] / counts[has]
            return out.reshape(n_span_bins, n_chord_bins), counts.reshape(n_span_bins, n_chord_bins)

        mag_grid, counts_grid = grid_mean(mag_m)
        u_grid, _ = grid_mean(u_m)
        v_grid, _ = grid_mean(v_m)
        r_grid, _ = grid_mean(r_m)

        valid = counts_grid >= min_count
        if not valid.any():
            return []

        threshold = np.nanpercentile(np.where(valid, mag_grid, np.nan), magnitude_percentile)

        d_span = span_centers[1] - span_centers[0]
        d_chord = chord_centers[1] - chord_centers[0]

        points = []

        for i in range(1, n_span_bins - 1):
            for j in range(1, n_chord_bins - 1):

                if not valid[i, j] or mag_grid[i, j] > threshold:
                    continue

                neighborhood_valid = valid[i - 1:i + 2, j - 1:j + 2]
                if not neighborhood_valid.all():
                    continue

                neighborhood_mag = mag_grid[i - 1:i + 2, j - 1:j + 2]
                if mag_grid[i, j] > neighborhood_mag.min():
                    continue

                du_dchord = (u_grid[i, j + 1] - u_grid[i, j - 1]) / (2 * d_chord)
                du_dspan = (u_grid[i + 1, j] - u_grid[i - 1, j]) / (2 * d_span)
                dv_dchord = (v_grid[i, j + 1] - v_grid[i, j - 1]) / (2 * d_chord)
                dv_dspan = (v_grid[i + 1, j] - v_grid[i - 1, j]) / (2 * d_span)

                jac = np.array([[du_dchord, du_dspan], [dv_dchord, dv_dspan]])
                eigvals = np.linalg.eigvals(jac)

                if np.any(np.abs(eigvals.imag) > 1e-12):
                    kind = 'focus'
                elif np.sign(eigvals[0].real) == np.sign(eigvals[1].real):
                    kind = 'node'
                else:
                    kind = 'saddle'

                points.append({
                    'span': span_centers[i], 'chord': chord_centers[j], 'r': r_grid[i, j],
                    'cf_mag': mag_grid[i, j], 'kind': kind,
                })

        return points

    def plot_critical_points(self, ax, points, marker_map=None, colors=None,
                              marker_size: float = 110, label: bool = True):

        '''
        Overlay critical_points()'s classified critical points - a
        distinct marker AND color per kind, so a vortex footprint
        ('focus') doesn't visually blend with an ordinary separation/
        attachment 'node' or 'saddle'.

        Parameters
        ----------
        ax : matplotlib Axes
        points : list[dict]
            critical_points()'s return value.
        marker_map, colors : dict, optional
            {'node': ..., 'saddle': ..., 'focus': ...} - matplotlib
            marker/color per kind. Defaults to a circle/x/star in
            blue/black/magenta.
        label : bool
            Add a legend entry for each kind present.
        '''

        marker_map = marker_map or {'node': 'o', 'saddle': 'x', 'focus': '*'}
        colors = colors or {'node': 'blue', 'saddle': 'black', 'focus': 'magenta'}

        for kind in ('node', 'saddle', 'focus'):
            sel = [p for p in points if p['kind'] == kind]
            if not sel:
                continue
            # 'x' has no fillable face - matplotlib warns (harmlessly) if an
            # edgecolor is passed for it, so only unfilled markers skip it.
            edge_kwargs = {} if marker_map[kind] == 'x' else {'edgecolors': 'white', 'linewidths': 0.8}
            ax.scatter([p['span'] for p in sel], [p['chord'] for p in sel], s=marker_size,
                       marker=marker_map[kind], color=colors[kind], label=kind if label else None,
                       zorder=6, **edge_kwargs)

    def save_critical_points(self, points, filepath: str):

        '''
        Write critical_points()'s results to a plain-text file: span,
        chord (physical, meters, raw Cartesian - see critical_points()'s
        Returns), r, cf_mag, kind - tab-separated with a header line.
        '''

        with open(filepath, 'w') as f:
            f.write('span_m\tchord_m\tr_m\tcf_mag\tkind\n')
            for p in points:
                f.write(f"{p['span']:.6f}\t{p['chord']:.6f}\t{p['r']:.6f}\t{p['cf_mag']:.6f}\t{p['kind']}\n")

    def poincare_index(self, points):

        '''
        Sum of Poincare indices over a critical_points() result: +1 per
        'node' or 'focus', -1 per 'saddle' (the standard index for each
        type - a node/focus is where the vector field's direction winds
        around once in the SAME sense as you walk around it; a saddle
        winds around once in the OPPOSITE sense - see README.md's
        "Theory" section under "Vortex-footprint critical points").

        The Poincare-Hopf theorem says this sum equals the Euler
        characteristic of the surface the vector field lives on, IF that
        surface is closed (no boundary) - e.g. 2 for a full blade's skin
        stitched into one sphere-like closed surface. critical_points()
        here runs on an OPEN patch (one surface - Upper or Lower -
        further cropped by span_min/span_max), NOT a closed surface, so
        there's no reason this sum should come out to 2 on this kind of
        selection - it's reported as a diagnostic number (and a way to
        sanity-check critical_points() against itself, e.g. across a
        resolution change), not a pass/fail check, unless a genuinely
        closed-surface analysis is built on top of this later.

        Returns
        -------
        int
        '''

        n_node = sum(1 for p in points if p['kind'] == 'node')
        n_focus = sum(1 for p in points if p['kind'] == 'focus')
        n_saddle = sum(1 for p in points if p['kind'] == 'saddle')

        return n_node + n_focus - n_saddle

    def _annotate_poincare_index(self, ax, points):

        '''Text-box annotation of poincare_index(points) - see that method and friction_lines()'s show_critical_points_index.'''

        index = self.poincare_index(points)
        note = ' (closed-surface value)' if index == 2 else ''
        ax.text(0.02, 0.02, f'$N+F-S = {index}${note}', transform=ax.transAxes, fontsize=11,
                va='bottom', ha='left', bbox=dict(facecolor='white', edgecolor='black', alpha=0.85))

    def friction_lines(self, surface=('Upper', 'Lower'), frame: int = None, span_min: float = None,
                        span_max: float = None, cf_clip_percentile: float = 99, n_arrows: int = 2000,
                        marker_size: float = 1, cmap: str = 'cividis', cbar_label: str = r'$C_f$ [-]',
                        show_separation_line: bool = False, separation_line_kwargs: dict = None,
                        show_migration_line: bool = False, migration_line_kwargs: dict = None,
                        show_critical_points: bool = False, critical_points_kwargs: dict = None,
                        show_critical_points_index: bool = False,
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
        show_separation_line : bool
            If True, compute separation_line() for each surface plotted
            and overlay its crossings (see that method and
            plot_separation_line()) - separation in one color,
            reattachment in another, with a legend entry for each.
        separation_line_kwargs : dict, optional
            Forwarded to separation_line() (n_span_bins, n_chord_bins,
            chord_percentile, reverse_chord) - frame/span_min/span_max
            are already taken from this call's own arguments, so don't
            repeat them here. In particular, pass the SAME reverse_chord
            you'd use with plot_cf_radii() on this case, or the overlay
            will disagree with cf_at_radii()'s own idea of which end is
            the leading edge.
        show_migration_line, migration_line_kwargs : bool, dict
            Same pattern as show_separation_line/separation_line_kwargs,
            for migration_line() instead (spanwise migration reversal,
            not chordwise separation - see that method).
        show_critical_points, critical_points_kwargs : bool, dict
            Same pattern, for critical_points() (vortex-footprint/
            topology critical points) - note critical_points() has no
            reverse_chord (works in raw physical chord, not x/c).
        show_critical_points_index : bool
            If True (and show_critical_points is also True), annotate
            each axes with poincare_index()'s N+F-S count in a text box -
            see that method's docstring for what it means and why it's
            not expected to equal 2 on this kind of open, cropped
            selection (only a genuinely closed surface must satisfy that).

        Returns
        -------
        (fig, axes)
            axes is a single Axes if surface is a single string, otherwise
            a list of Axes in the same order as surface.
        '''

        if self.rho_ref is None or self.rpm is None:
            raise ValueError("rho_ref and rpm must be set (in __init__) to compute Cf.")

        surfaces = (surface,) if isinstance(surface, str) else tuple(surface)
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

            any_overlay = False

            if show_separation_line:
                sep_points = self.separation_line(surface=surf, frame=frame, span_min=span_min,
                                                   span_max=span_max, **(separation_line_kwargs or {}))
                self.plot_separation_line(ax, sep_points)
                any_overlay = any_overlay or bool(sep_points)

            if show_migration_line:
                mig_points = self.migration_line(surface=surf, frame=frame, span_min=span_min,
                                                  span_max=span_max, **(migration_line_kwargs or {}))
                self.plot_migration_line(ax, mig_points)
                any_overlay = any_overlay or bool(mig_points)

            if show_critical_points:
                crit_points = self.critical_points(surface=surf, frame=frame, span_min=span_min,
                                                    span_max=span_max, **(critical_points_kwargs or {}))
                self.plot_critical_points(ax, crit_points)
                any_overlay = any_overlay or bool(crit_points)

                if show_critical_points_index:
                    self._annotate_poincare_index(ax, crit_points)

            if any_overlay:
                ax.legend(loc='upper right')

            ax.set_ylabel('chord [m]')
            ax.set_aspect('equal')

        axes[-1].set_xlabel('span [m]')
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, (axes[0] if len(axes) == 1 else list(axes))
