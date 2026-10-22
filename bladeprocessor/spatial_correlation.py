'''
Spatial two-point correlation of a time-resolved SCALAR surface field -
the SPATIAL analogue of bladeprocessor/convergence.py's autocorrelation()/
integral_timescale() (temporal correlation structure, lag measured in
TIME), now asking the same kind of question with lag measured in SPACE:
how far apart can two surfels be before their own fluctuating time
series stop being meaningfully related to each other at all. Standalone
and class-agnostic, same (n_frames, n_points) convention as
bladeprocessor/pod.py's pod() - fed by whichever class's own
cf_time_series()/cp_time_series() is appropriate, exactly like every
other module in this project.

Deliberately NOT a POD- or DMD-based tool: unlike bladeprocessor/pod.py
or bladeprocessor/dmd.py, which both operate on a field's own reduced,
energy-ranked basis, this module asks its question of the RAW field
directly - the coherence length answered here describes the physical
size of whatever structure is present without first assuming how many
POD modes are needed to represent it, which makes it a useful,
independent cross-check of a POD/DMD-based finding rather than a
restatement of it (e.g. "does the coherence length estimated here agree
with the characteristic in-plane extent of the dominant POD mode's own
spatial pattern").

Correlating one point against every other, rather than every pair of
points against each other (an n_points x n_points problem, intractable
at this project's real surfel counts), is the practical restriction that
makes this tractable - see two_point_correlation()'s own `ref_index`
parameter for how that one point should be chosen.
'''

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


def two_point_correlation(data, radius, chord, ref_index: int = None,
                           n_bins: int = 30, max_distance: float = None):

    '''
    Pearson correlation coefficient, at zero time lag, between ONE
    reference surfel's own time series and EVERY other surfel's own time
    series, then binned by physical separation distance from that
    reference point - the spatial correlation function R(r) whose decay
    with distance sets a coherence length, the same role
    bladeprocessor.convergence.autocorrelation()'s rho(s) plays for a
    temporal lag (see that function's own docstring for the same
    "decays fast = little memory/short structure, decays slowly =
    persistent/large structure" reading, now in space rather than time).

    The correlation coefficient already subtracts each series' own mean
    internally (the standard Pearson definition), so `data` can be
    passed exactly as already extracted for pod() - the SAME raw,
    per-frame (n_frames, n_points) array, with NO separate mean-removal
    step needed first.

    Choosing the reference point - `ref_index`:
        None (default): the point with the largest fluctuation RMS
            (std over time) is used automatically. The single most
            energetic point in the field is the natural anchor for a
            coherence-length estimate: a reference point sitting in a
            quiet, noise-dominated part of the field would measure
            mostly how far UNCORRELATED NOISE decorrelates, not the
            actual coherent structure's own size.
        An explicit int: use this when a specific location is already
            of interest - in particular, the peak of the POD mode
            already identified as the dominant coherent structure (see
            bladeprocessor.pod.pod()'s own 'modes' key),
            `ref_index = np.argmax(np.abs(pod_result['modes'][0]))`,
            anchors the correlation length specifically on THAT
            structure rather than on whichever point happens to have
            the largest raw RMS (not necessarily the same point, if the
            field's overall RMS is dominated by broadband turbulence
            rather than by the identified structure itself - exactly the
            situation already observed in this project's own Cf/Cp POD
            energy spectra, where the first mode alone does not fully
            dominate). Calling this function several times, with
            `ref_index` set to a few different stations along the same
            structure (e.g. a handful of chordwise locations along the
            reattachment line), is also how to check whether the
            coherence length itself GROWS along the direction a
            structure convects - the direct spatial signature of the
            vortex-pairing cascade already discussed in this project
            (K-H rollers merging and growing in size as they convect
            downstream).

    Parameters
    ----------
    data : array-like, shape (n_frames, n_points)
        Scalar surface field time series, one column per surfel - same
        convention as pod()'s own `data` parameter.
    radius, chord : array-like, shape (n_points,)
        Physical radius [m] and chord [m] of each point, same convention
        and SAME point order as bladeprocessor.pod.plot_pod_mode()'s own
        radius/chord parameters (e.g. FrictionLines._radius(surface)/
        ..._span_chord(surface)'s second return value, already masked to
        whatever region `data` itself was extracted from).
    n_bins : int
        Number of equal-width distance bins the raw, unstructured
        (distance, correlation) scatter is averaged into before
        integrating - see `distance_bins`/`correlation_binned` below.
        A raw CFD surface mesh has no regular spacing, so a handful of
        points at nearly the same distance from the reference point can
        otherwise scatter considerably even at small separations; binning
        trades resolution for a smoother, more reliable R(r) curve, the
        same trade Welch's method makes in frequency (see
        bladeprocessor.convergence.periodogram()'s own n_chunk).
    max_distance : float, optional
        Largest separation distance to bin up to - defaults to the
        largest distance actually present in the selected region. Set
        this explicitly to a smaller value to concentrate all `n_bins`
        within the near-field distances actually expected to carry a
        real correlation, rather than spreading them out to the full,
        mostly-decorrelated far field.

    Returns
    -------
    dict with keys:
        ref_index : int
        ref_position : tuple (radius, chord) of the reference point.
        raw_distance : np.ndarray, shape (n_points,)
            Euclidean distance from the reference point to every point
            (0 at the reference point itself).
        raw_correlation : np.ndarray, shape (n_points,)
            Per-point Pearson correlation coefficient with the reference
            point's own time series (exactly 1.0 at the reference point).
        distance_bins : np.ndarray, shape (n_bins,)
            Bin center distances.
        correlation_binned : np.ndarray, shape (n_bins,)
            Mean raw_correlation of every point falling in each distance
            bin - NaN where no point fell in that bin.
        distance_profile, correlation_profile : np.ndarray
            distance_bins/correlation_binned with any NaN (empty) bins
            dropped and the exact point (0, 1.0) prepended - the curve
            actually integrated below, and the one to plot (see
            plot_two_point_correlation()).
        length_scale : float
            Coherence length - the integral of correlation_profile from
            r=0 up to its first zero crossing (returning the full
            integral, to the end of the profile, if the correlation
            never crosses zero), the exact spatial analogue of
            bladeprocessor.convergence.integral_timescale()'s own
            "integrate rho(s) to its first zero crossing" definition -
            see that function's docstring for why stopping there, rather
            than integrating the full range, avoids accumulating the
            correlation estimate's own noise floor at large separation.
    '''

    data = np.asarray(data, dtype=float)
    radius = np.asarray(radius, dtype=float)
    chord = np.asarray(chord, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_frames, n_points), got shape {data.shape}")
    n_frames, n_points = data.shape
    if radius.shape != (n_points,) or chord.shape != (n_points,):
        raise ValueError(
            f"radius/chord must have shape ({n_points},) matching data's own n_points - "
            f"got radius {radius.shape}, chord {chord.shape}."
        )

    if ref_index is None:
        ref_index = int(np.argmax(data.std(axis=0)))
    elif not (0 <= ref_index < n_points):
        raise ValueError(f"ref_index={ref_index} out of range for {n_points} points.")

    ref = data[:, ref_index]
    ref_c = ref - ref.mean()
    ref_std = ref_c.std()
    if ref_std == 0:
        raise ValueError(
            f"Reference point (index {ref_index}) has zero variance over time - "
            "correlation is undefined. Pick a different ref_index."
        )

    data_c = data - data.mean(axis=0)
    data_std = data_c.std(axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        raw_correlation = (ref_c[:, None] * data_c).mean(axis=0) / (ref_std * data_std)
    raw_correlation[data_std == 0] = 0.0  # a zero-variance (e.g. degenerate) point carries no correlation
    raw_correlation[ref_index] = 1.0  # exact, not whatever floating-point value the division above happens to give

    raw_distance = np.sqrt((radius - radius[ref_index]) ** 2 + (chord - chord[ref_index]) ** 2)

    if max_distance is None:
        max_distance = raw_distance.max()
    edges = np.linspace(0, max_distance, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_idx = np.clip(np.digitize(raw_distance, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    has_data = counts > 0

    sums = np.zeros(n_bins)
    np.add.at(sums, bin_idx, raw_correlation)
    correlation_binned = np.full(n_bins, np.nan)
    correlation_binned[has_data] = sums[has_data] / counts[has_data]

    # The reference point's own exact (distance=0, correlation=1) point is
    # prepended explicitly, rather than relying on however many OTHER
    # points happen to land in the very first bin (typically very few,
    # since distance=0 is a single point by construction) to pin this
    # endpoint - mirrors autocorrelation()'s own rho[0] == 1.0 exact
    # convention.
    distance_profile = np.concatenate(([0.0], centers[has_data]))
    correlation_profile = np.concatenate(([1.0], correlation_binned[has_data]))

    sign_changes = np.where(np.diff(np.sign(correlation_profile)) < 0)[0]  # first + -> - crossing
    cutoff = sign_changes[0] + 1 if len(sign_changes) > 0 else len(correlation_profile) - 1
    length_scale = np.trapz(correlation_profile[:cutoff + 1], distance_profile[:cutoff + 1])

    return {
        'ref_index': ref_index,
        'ref_position': (radius[ref_index], chord[ref_index]),
        'raw_distance': raw_distance,
        'raw_correlation': raw_correlation,
        'distance_bins': centers,
        'correlation_binned': correlation_binned,
        'distance_profile': distance_profile,
        'correlation_profile': correlation_profile,
        'length_scale': length_scale,
    }


def plot_two_point_correlation(data, radius, chord, ref_index: int = None, n_bins: int = 30,
                                max_distance: float = None, ax=None, color: str = 'black',
                                savepath: str = None, dpi: int = 600):

    '''
    Plot two_point_correlation()'s binned R(r) (with the raw, unbinned
    per-surfel scatter underneath, for a sense of how much binning
    actually smoothed) against separation distance, with the integrated
    region (r=0 up to the first zero crossing) shaded and the resulting
    coherence length annotated - same style as
    bladeprocessor.convergence.plot_integral_timescale().

    Returns
    -------
    (fig, ax)
    '''

    result = two_point_correlation(data, radius, chord, ref_index=ref_index,
                                    n_bins=n_bins, max_distance=max_distance)
    d_profile, r_profile = result['distance_profile'], result['correlation_profile']

    sign_changes = np.where(np.diff(np.sign(r_profile)) < 0)[0]
    cutoff = sign_changes[0] + 1 if len(sign_changes) > 0 else len(r_profile) - 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.scatter(result['raw_distance'], result['raw_correlation'], s=4, color=color,
               alpha=0.15, linewidths=0)
    ax.plot(d_profile, r_profile, 'o-', color=color, linewidth=1.5, markersize=4,
            label=f'Binned ({n_bins} bins)')
    ax.fill_between(d_profile[:cutoff + 1], r_profile[:cutoff + 1], 0, color=color, alpha=0.15,
                     label='Integrated region')
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Separation distance $r$ [m]')
    ax.set_ylabel('Correlation coefficient [-]')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

    note = (
        f"$L_c$ = {result['length_scale']:.4g} m\n"
        f"Ref.: $(r,c)$ = ({result['ref_position'][0]:.4g}, {result['ref_position'][1]:.4g})"
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=14, va='bottom', ha='left',
            zorder=10, bbox=dict(facecolor='white', edgecolor='black', alpha=1.0))

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax
