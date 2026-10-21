'''
Proper Orthogonal Decomposition (POD) of a time-resolved SCALAR surface
field - a class-agnostic pair of tools (the decomposition itself, and its
visualization) built to work identically whether the field being
decomposed is a wall-shear/Cf quantity from FrictionLines or a
static-pressure/Cp quantity from SurfaceVariable, since the underlying
linear algebra doesn't care which physical quantity it's applied to -
only that the data arrives as a plain (n_frames, n_points) array, the
SAME shape convention already used by FrictionLines.cf_time_series()/
SurfaceVariable.variable_time_series()/cp_time_series(). Deliberately
standalone, not a method on either class, for the same reason
bladeprocessor/convergence.py's functions are standalone: one
implementation, fed by whichever class's own data-extraction method is
appropriate, rather than duplicating the same SVD twice.

Background: POD decomposes a fluctuating field q(x,t) (here, x running
over surfels, t over frames) into a sum of spatial modes phi_k(x), each
carrying its own time-dependent amplitude a_k(t),

    q(x,t) ~= sum_k a_k(t) * phi_k(x) ,

with the modes ranked by how much of the total fluctuation ENERGY
(variance) each one accounts for, and with no assumption about frequency
content at all (unlike a Fourier/harmonic decomposition - see
StripForces.harmonics()'s own docstring - POD modes are generally NOT
single-frequency; they are simply the most *energetically* efficient,
mutually orthogonal basis for representing the recorded fluctuation).
This makes POD the right tool for asking "what is the single most
repeatable SPATIAL pattern of unsteadiness, and how much of the total
fluctuation does it explain" - as opposed to "what frequency content is
present" (that question belongs to harmonics()/periodogram() instead).

IMPORTANT - memory, consistent with this project's own history (see
converters/snc_reader.py's and FrictionLines/StripForces's own
constructor-level span_min/span_max cropping, both added after a real
OOM on an uncropped whole-rotor case): the data matrix handed to pod()
has one column per surfel, so a full, uncropped surface can easily reach
hundreds of thousands (or, on a DNS-resolution case, tens of millions)
of columns. For SurfaceVariable, use variable_time_series_region()/
cp_time_series_region() (NOT variable_time_series()/cp_time_series()
plus a post-hoc column crop, as an earlier version of this docstring's
own examples did) - the region crop is applied PER FRAME-CHUNK, during
a streamed read, so the full (n_frames, n_points) array is never held
in memory at once (confirmed a real OOM on a DNS-resolution case
otherwise - see those methods' own docstrings). The OUTPUT itself is
still held in full, though, so a region that's still too large for
THAT to fit needs to be genuinely smaller - see "region" below for how
to pick one (restricted to wherever the structure of interest actually
is, not the whole blade).

pod() is applied to ONE surface at a time (e.g. the suction/Upper side
alone, if that is where the phenomenon of interest - a separation
bubble, a leading-edge-vortex-like structure - actually lives), matching
how cf_time_series()/cp_time_series() are already surface-specific; it
is not meant to combine Upper and Lower into a single call. Typical
usage, either data source - region : restrict to whatever span window
actually contains the structure of interest BEFORE calling pod() (both
for memory, see above, and because a structure diluted across a huge,
mostly-irrelevant region competes poorly, in pure energy terms, against
whatever else is happening elsewhere on the blade). This is not
necessarily the tip - a structure identified from the friction-line/
critical-point analysis may be centered well inboard, so the region
should be set from wherever that analysis actually located it, not
assumed in advance:

    # Cf (FrictionLines), suction (Upper) side only:
    span, chord = fl._span_chord('Upper')     # span: for the region mask
    radius = fl._radius('Upper')              # radius: for the plot's x-axis (see plot_pod_mode())
    region = (span > 0.05) & (span < 0.10)    # wherever the structure actually is
    cf_data = fl.cf_time_series(surface='Upper', component=None)[:, region]
    result = pod(cf_data, n_modes=10)
    plot_pod_energy(result, savepath='pod_energy_cf.png')  # look here first, decide which mode(s) matter
    plot_pod_mode(radius[region], chord[region], result, mode_index=0,
                  savepath='pod_mode1_cf.png')
    plot_pod_temporal_coefficient(result, mode_index=0, dt=dt,
                                   savepath='pod_coeff_mode1_cf.png')

    # Cp (SurfaceVariable) - same pattern, same functions:
    span, chord = sv._span_chord('Upper')
    radius = sv._radius('Upper')
    region = (span > 0.05) & (span < 0.10)
    cp_data = sv.cp_time_series_region(surface='Upper', region_mask=region)  # streamed - see memory note above
    result = pod(cp_data, n_modes=10)
    plot_pod_mode(radius[region], chord[region], result, mode_index=0,
                  unit_label='$C_p$ mode amplitude [-]',
                  savepath='pod_mode1_cp.png')
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import csd, coherence

from bladeprocessor.convergence import periodogram

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18
})


def pod(data, n_modes: int = None, subtract_mean: bool = True, weights=None):

    '''
    Proper Orthogonal Decomposition of a time-resolved scalar field via
    the "method of snapshots": X's own (n_frames, n_frames) Gram matrix
    is formed and eigendecomposed explicitly, rather than handing X
    itself (n_frames, n_points) to a dense SVD routine. An earlier
    version of this function called np.linalg.svd(X, full_matrices=False)
    directly - mathematically equivalent and, in FLOP terms, just as
    efficient when n_points >> n_frames (always true for a CFD surface) -
    but confirmed, via a controlled reproduction on a real case, to
    SEGFAULT (no Python traceback at all) at this project's actual data
    scale: a (614-682, several million) matrix handed to np.linalg.svd
    reliably crashed inside libflame.so (this cluster's LAPACK), and did
    so NON-DETERMINISTICALLY depending on which compute node it landed on
    for the identical shape - i.e. this isn't a clean "too big, always
    fails" threshold to design around by shrinking the POD region; it's
    a latent bug in the dense SVD driver itself at this scale, and a run
    that happens to succeed once is not good evidence it'll succeed
    again. Forming the Gram matrix via a plain BLAS matmul (X @ X.T) and
    eigendecomposing that small (n_frames, n_frames) result instead never
    invokes that code path - confirmed safe at the same real-data scale
    that reliably crashed the direct-SVD version.

    data is kept in whatever floating dtype it's already given (NOT
    forced to float64) - confirmed a real OOM on another real case
    otherwise: the (n_frames, n_points) POD region materialized by the
    caller is already float32 straight off an HDF5 'f4' dataset, and an
    unconditional float64 upcast briefly held both the original and a
    second, doubled-size copy alive at once, on top of whatever else was
    already resident. Every (n_frames, n_points)-sized array in this
    function (X, Vt) stays in that same native dtype throughout; only
    the small (n_frames, n_frames) eigenvalue problem runs in float64,
    for better-conditioned eigenvalues at essentially no memory cost.
    Checked empirically against the original full-float64 path on
    realistic (spatially-correlated, not i.i.d. noise) synthetic data:
    singular values/energy fractions/reconstruction all agree to within
    ~0.1-0.3% relative error - negligible for POD's own purpose (energy
    ranking, mode shapes), and no worse than the precision already lost
    when the source file itself was written as float32 to begin with.

    subtract_mean : bool
        If True (default), each point's own time-mean is removed before
        the SVD, so the modes describe the FLUCTUATION about the mean
        field, not the mean field itself - the steady/mean part is a
        single, already-understood quantity (see
        FrictionLines.cf()/SurfaceVariable.cp()'s own stat='mean'), not
        something a modal decomposition of the UNSTEADY signal should
        spend a mode on. The removed mean is returned (key 'mean') for
        reference; add it back explicitly if an absolute-level
        reconstruction, not just the fluctuating part, is wanted (same
        convention as StripForces.reconstruct_from_harmonics()'s own
        mean-handling).
    weights : array-like, shape (n_points,), optional
        Per-point weights (typically each surfel's own AREA) for a
        physically energy-correct decomposition when the mesh is
        non-uniform - unweighted POD (the default, weights=None)
        implicitly treats every point as equally important regardless of
        how much physical surface area it actually represents, which
        biases the "energy" ranking toward whatever part of the mesh
        happens to be most finely resolved (e.g. near the leading edge),
        not necessarily where the physical fluctuation energy actually
        is. Passing the surfel areas (e.g. from Geometry/<surface>/Area,
        where available) corrects for this: internally, the data is
        scaled by sqrt(weights) before the SVD (so that a weighted inner
        product becomes an ordinary one), and the resulting spatial
        modes are rescaled back by 1/sqrt(weights) afterward, so 'modes'
        and 'coefficients' are returned in the same physical units
        either way - only the relative ranking/energy split changes.

    Returns
    -------
    dict with keys:
        modes : np.ndarray, shape (k, n_points)
            Spatial modes phi_k(x), one per row, orthonormal in the
            (possibly weighted) inner product - phi_k . phi_l = delta_kl
            unweighted, or sum_i weights_i phi_k(i) phi_l(i) = delta_kl
            if weights was given.
        coefficients : np.ndarray, shape (n_frames, k)
            Temporal coefficients a_k(t), one per column - the field's
            reconstruction (fluctuating part only if subtract_mean) is
            sum_k coefficients[:,k:k+1] @ modes[k:k+1,:].
        singular_values : np.ndarray, shape (k,)
        energy_fraction : np.ndarray, shape (k,)
            S_k^2 / sum(S^2) - the fraction of the total fluctuation
            energy captured by mode k alone.
        cumulative_energy : np.ndarray, shape (k,)
            Running sum of energy_fraction - how much of the total is
            captured by the first k modes together.
        mean : np.ndarray, shape (n_points,), or None
            The per-point time-mean removed before the SVD, if
            subtract_mean was True; None otherwise.
        n_modes_total : int
            min(n_frames, n_points) - how many modes exist in total
            before any n_modes truncation (the rank of the decomposition).
    '''

    # Kept in whatever floating dtype the caller already has (NOT forced
    # to float64) - the one thing that actually matters for memory here
    # is X/data/Vt, each (n_frames, n_points), which for a real POD
    # region on a large case can be hundreds of GB on its own (confirmed
    # a real OOM on a real case: forcing a float64 upcast of an already-
    # float32 array - e.g. straight off an HDF5 'f4' dataset - briefly
    # held BOTH the original and the doubled-size copy alive at once,
    # on top of whatever else was already resident). An integer input
    # still gets promoted to float64 (there's no meaningful "native
    # dtype" to preserve there, and POD's arithmetic needs floats anyway).
    data = np.asarray(data)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_frames, n_points), got shape {data.shape}")
    n_frames, n_points = data.shape
    if n_frames < 2:
        raise ValueError(f"Need at least 2 frames for a meaningful POD - got {n_frames}.")

    mean_field = data.mean(axis=0) if subtract_mean else None
    X = (data - mean_field[None, :]) if subtract_mean else data.copy()

    sqrt_w = None
    if weights is not None:
        # Cast to X's own dtype, not a fixed float64 - mixing float32 X
        # with float64 weights would silently promote the big X * sqrt_w
        # multiply back to float64 (numpy's usual type-promotion rule),
        # re-introducing the exact memory blowup the dtype handling above
        # exists to avoid.
        weights = np.asarray(weights, dtype=X.dtype)
        if weights.shape != (n_points,):
            raise ValueError(f"weights must have shape ({n_points},), got {weights.shape}.")
        if np.any(weights < 0):
            raise ValueError("weights (surfel areas) must be non-negative.")
        sqrt_w = np.sqrt(weights)
        X = X * sqrt_w[None, :]

    # Method of snapshots: eigendecompose the small (n_frames, n_frames)
    # Gram matrix instead of calling np.linalg.svd(X, ...) directly on the
    # full (n_frames, n_points) matrix - see this function's own
    # docstring for why (a confirmed SVD-driver segfault at this
    # project's real data scale). eigh (not eig) because X @ X.T is
    # exactly symmetric by construction, and its eigenvalues are the
    # squared singular values of X - so they're mathematically
    # guaranteed non-negative; any small negative values that show up
    # are floating-point noise, clipped away before the sqrt below.
    # The (n_frames, n_points) matmul forming it runs in X's own
    # (possibly float32) dtype - that's the expensive part, and it's the
    # same accumulation that already happens for any other operation on
    # X regardless of precision. The RESULT, gram, is only (n_frames,
    # n_frames) - trivially small even for a huge POD region - so it's
    # upcast to float64 before eigh for better-conditioned eigenvalues;
    # this costs nothing memory-wise (a few hundred MB at most) and loses
    # nothing, since there's no extra precision to preserve in a matrix
    # this small anyway.
    gram = (X @ X.T).astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    U = eigvecs[:, order]
    S = np.sqrt(eigvals)
    n_modes_total = len(S)

    # Spatial modes (V^T = S^-1 U^T X) - a cheap (n_frames, n_frames) @
    # (n_frames, n_points) matmul, not another dense SVD/eigendecomposition.
    # U/S are cast down to X's own dtype for this one step - the only
    # arithmetic here touching a (n_frames, n_points)-sized array - so
    # the result (Vt) stays in X's dtype instead of silently promoting
    # back to float64 via numpy's float32-op-float64 promotion rule
    # (U/S are tiny either way, so this loses nothing that matters: U's
    # entries are bounded orthonormal-matrix values, and S's eigh-based
    # float64 precision was only ever needed for the SMALL eigenvalue
    # problem itself, not for this one multiply).
    # A ~zero singular value (only possible if two snapshots are exactly
    # linearly dependent, e.g. a duplicated frame) would make that one
    # row's direction ill-defined, not wrong - left as all-zero rather
    # than dividing by ~0 and amplifying noise into a meaningless mode.
    #
    # k modes, not all n_modes_total (= n_frames, generically) - Vt's
    # FULL size would be (n_frames, n_points), the same size as X itself
    # (confirmed a real OOM on a real case: with n_modes=10 requested but
    # n_frames~5300, materializing all ~5300 spatial modes before
    # truncating down to 10 at the very end briefly held X, data, AND a
    # full-size Vt alive at once - three X-sized arrays, not two). Only
    # the k actually-requested columns of U/rows of S are ever used below,
    # so Vt is sized (k, n_points) from the start - for k=10, utterly
    # negligible next to X regardless of n_points. n_modes=None (every
    # mode genuinely wanted) is the one case where this is still exactly
    # as expensive as before - an explicit, deliberate ask, not a hidden
    # default.
    k = n_modes_total if n_modes is None else min(n_modes, n_modes_total)

    nonzero_k = S[:k] > 0
    U_k = U[:, :k][:, nonzero_k].astype(X.dtype)
    S_k = S[:k][nonzero_k].astype(X.dtype)
    Vt = np.zeros((k, n_points), dtype=X.dtype)
    Vt[nonzero_k] = (U_k.T @ X) / S_k[:, None]

    if sqrt_w is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            Vt = Vt / sqrt_w[None, :]
        Vt[:, weights == 0] = 0.0  # a zero-area/degenerate point carries no mode content

    energy_fraction_full = S ** 2 / np.sum(S ** 2)

    return {
        'modes': Vt,
        'coefficients': U[:, :k] * S[None, :k],
        'singular_values': S[:k],
        'energy_fraction': energy_fraction_full[:k],
        'cumulative_energy': np.cumsum(energy_fraction_full)[:k],
        'mean': mean_field,
        'n_modes_total': n_modes_total,
    }


def plot_pod_mode(radius, chord, pod_result: dict, mode_index: int = 0,
                   cmap: str = 'RdBu_r', marker_size: float = 1, value_clip_percentile: float = 99,
                   unit_label: str = None, figsize: tuple = None, ax=None, savepath: str = None,
                   dpi: int = 600, radius_full=None, chord_full=None, background_color: str = '0.85'):

    '''
    ONE POD spatial mode, plotted the same way this project already
    plots any other surface field on the actual blade geometry - see
    SurfaceVariable.plot_variable_surface()/FrictionLines.friction_lines():
    a raw per-surfel scatter (radius vs. chord, both in physical units),
    deliberately NOT interpolated or gridded first (that earlier approach
    elsewhere in this project was both slow and introduced interpolation
    artifacts - a raw point cloud already reads as a continuous field at
    typical surfel densities - see the note on apparent "fringing" at low
    point density below). This is the single most direct way to answer
    "where on the real blade is this mode" - the same question
    plot_variable_surface() answers for a raw physical field, now asked
    of a POD mode instead.

    One mode per call, by design, so each gets a full-sized, individually
    readable plot rather than a small panel in a crowded grid - call this
    once per mode actually worth a closer look (typically identified
    first from plot_pod_energy()'s scree plot), not in a batch loop over
    every mode that exists.

    One difference from plot_variable_surface()'s own color scaling:
    there, the clip range is the plain two-sided percentile of the
    (generally asymmetric) physical field. Here, since a mode has no
    inherent reason to be skewed toward one sign over the other (only
    its product with its own time-varying coefficient does), the color
    scale is instead forced SYMMETRIC about zero - vmax is the
    value_clip_percentile of |mode|, and the colormap spans [-vmax,
    vmax] - so equal color intensity reliably means equal contribution
    magnitude regardless of sign.

    A note on "fringing"/speckle: whether this looks like a smooth,
    continuous field or a scatter of disconnected dots depends entirely
    on how DENSE the point cloud is relative to the marker size, exactly
    as for any other raw-scatter surface plot in this project - a coarse
    synthetic test grid (a few thousand points) will show visible gaps
    that a real CFD surface mesh (tens to hundreds of thousands of
    surfels) will not, at the same marker_size. Separately, and not an
    artifact at all: a genuinely noise-dominated mode (an uninterpretable
    one, typically far down the energy ranking - see plot_pod_energy())
    WILL look speckled regardless of point density, because it IS mostly
    uncorrelated noise from one point to the next - that speckle is the
    correct, honest rendering of what that particular mode actually is,
    not a plotting problem to fix.

    radius, chord : array-like, shape (n_points,)
        Physical radius [m] (e.g. FrictionLines._radius(surface) or
        SurfaceVariable._radius(surface)) and chord [m]
        (..._span_chord(surface)'s second return value) - same
        convention and units as plot_variable_surface()'s own x/y axes.
        Same n_points, same point ORDER, as the data pod() was computed
        from - whatever region selection was applied before calling
        pod() must be applied identically here.
    radius_full, chord_full : array-like, shape (n_points_full,), optional
        The SAME blade surface's full, unrestricted radius/chord (e.g.
        before any pod_span_min/pod_span_max/pod_chord_min/pod_chord_max
        crop) - when given, plotted FIRST as a neutral grey backdrop
        (background_color), with the actual mode scatter (radius/chord
        above, i.e. the cropped POD region) drawn on top. Gives the
        blade's real physical context for a POD deliberately restricted
        to a SMALL region (e.g. for memory reasons on a very large case -
        see SurfaceVariable.variable_time_series_region()'s own
        docstring) - without this, a narrow region plotted alone leaves
        the rest of the axes blank once set_aspect('equal') stretches
        them to fit figsize, which reads as missing/corrupted data rather
        than "deliberately not computed here". Omit both (default) for
        the previous behavior - just the region itself, no backdrop -
        e.g. when the POD already covers the whole blade and a backdrop
        would add nothing.
    background_color : str
        Matplotlib grayscale shorthand ('0.0' black - '1.0' white) for
        the backdrop scatter, when radius_full/chord_full are given.
    mode_index : int
        0-indexed - mode_index=0 is the single most energetic mode.
    value_clip_percentile : float
        See plot_variable_surface()'s parameter of the same name - caps
        the color scale so a handful of extreme surfels don't wash out
        the rest; set to 100 to disable.
    unit_label : str, optional
        Colorbar label - defaults to a generic 'Mode amplitude [-]';
        pass something like '$C_p$ mode amplitude [-]' or
        '$C_f$ mode amplitude [-]' for the quantity actually decomposed.
    figsize : tuple, optional
        (9, 5) by default - same convention as
        FrictionLines.friction_lines()/SurfaceVariable.plot_variable_surface()'s
        own figsize parameter: a plain fixed default, with the CALLER
        responsible for passing a region-appropriate value (e.g. this
        project's own per-case blade_figsize, already threaded through
        to every other plot in pressure_manager.py/skin_friction_manager.py)
        when the default doesn't suit a particular region. An earlier
        version of this parameter tried to auto-compute a "data-matching"
        figsize instead - that broke down for a POD region cropped to a
        narrow spanwise slice (chord spanning several times more than
        the radius band itself, the opposite of a whole-blade plot),
        which also then threw off the colorbar/label layout tuned for a
        normally-proportioned figure. set_aspect('equal') below still
        renders the TRUE physical proportions correctly inside whatever
        figsize is given - a region whose own aspect doesn't match the
        chosen figsize will show as a plot that doesn't fill the full
        width or height of its axes box, which is expected, not a bug.

    Returns
    -------
    (fig, ax)
    '''

    radius = np.asarray(radius)
    chord = np.asarray(chord)
    modes = pod_result['modes']
    energy = pod_result['energy_fraction']

    if mode_index >= modes.shape[0]:
        raise ValueError(f"mode_index={mode_index} out of range - only {modes.shape[0]} mode(s) available.")
    if radius.shape != chord.shape or radius.shape[0] != modes.shape[1]:
        raise ValueError(
            f"radius/chord (shape {radius.shape}) must match pod_result['modes']'s own "
            f"n_points ({modes.shape[1]}) - same region selection used for both."
        )

    mode = modes[mode_index]
    vmax = np.percentile(np.abs(mode), value_clip_percentile) if mode.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    mode_clipped = np.clip(mode, -vmax, vmax)

    if (radius_full is None) != (chord_full is None):
        raise ValueError("radius_full and chord_full must be given together.")

    if ax is None:
        if figsize is None:
            figsize = (9, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if radius_full is not None:
        # zorder=1, drawn before (so underneath) the actual mode scatter
        # below - plain context, not data, so it carries no colormap/
        # colorbar of its own.
        ax.scatter(np.asarray(radius_full), np.asarray(chord_full), s=marker_size,
                   color=background_color, linewidths=0, zorder=1)

    sc = ax.scatter(radius, chord, c=mode_clipped, s=marker_size, cmap=cmap, vmin=-vmax, vmax=vmax, zorder=2)
    ax.set_title(f'Mode {mode_index + 1} ({100 * energy[mode_index]:.1f}\\% energy)')
    ax.set_xlabel('$r$ [m]')
    ax.set_ylabel('chord [m]')
    ax.set_aspect('equal')
    # Horizontal, below the panel - matches this plot's own wide/short
    # shape (radius typically spans a much larger range than chord, same
    # as every other surface plot in this project) far better than a
    # vertical colorbar, which adds width without adding information.
    fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.18, aspect=30,
                 label=unit_label or 'Mode amplitude [-]')

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax


def plot_pod_energy(pod_result: dict, n_modes: int = None, ax=None, savepath: str = None, dpi: int = 600):

    '''
    Scree plot (per-mode energy fraction, log scale, left axis) with the
    cumulative energy overlaid (right axis) - the standard way to decide
    how many modes actually matter: a steep initial drop followed by a
    long, flat tail of comparably small contributions indicates a
    genuinely low-dimensional, energetically dominant coherent structure
    (the hoped-for outcome when checking a specific physical hypothesis,
    e.g. a leading-edge-vortex-like structure, against an objective,
    assumption-free energy ranking); a slowly, smoothly decaying spectrum
    with no clear "elbow" instead indicates broadband, high-dimensional
    content with no single dominant pattern.

    Returns
    -------
    (fig, (ax, ax_cumulative))
    '''

    energy = pod_result['energy_fraction']
    cumulative = pod_result['cumulative_energy']
    n_modes = len(energy) if n_modes is None else min(n_modes, len(energy))
    mode_idx = np.arange(1, n_modes + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.bar(mode_idx, 100 * energy[:n_modes], color='tab:blue')
    ax.set_yscale('log')
    ax.set_xlabel('POD mode')
    ax.set_ylabel('Energy fraction [\\%]', color='tab:blue')
    ax.set_xticks(mode_idx)
    ax.tick_params(axis='y', colors='tab:blue')
    ax.grid(True, which='both', axis='y', alpha=0.3)

    ax_cum = ax.twinx()
    ax_cum.plot(mode_idx, 100 * cumulative[:n_modes], 'o-', color='tab:red')
    ax_cum.set_ylabel('Cumulative energy [\\%]', color='tab:red')
    ax_cum.set_ylim(0, 105)
    ax_cum.tick_params(axis='y', colors='tab:red')

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, (ax, ax_cum)


def plot_pod_temporal_coefficient(pod_result: dict, mode_index: int = 0, dt: float = None,
                                   ax=None, savepath: str = None, dpi: int = 600):

    '''
    Time trace of a single POD mode's own temporal coefficient a_k(t) -
    useful for checking whether a spatially dominant mode is itself
    periodic (tied to the rotor's own rotation, in which case it should
    show up at the corresponding harmonic in a Fourier decomposition of
    a_k(t) too - see StripForces.harmonics(), applicable directly to this
    1D series), broadband/turbulent, or carrying its own, distinct
    characteristic frequency unrelated to blade passage.

    mode_index : int
        0-indexed - mode_index=0 is the single most energetic mode.

    Returns
    -------
    (fig, ax)
    '''

    coeff = pod_result['coefficients']
    if mode_index >= coeff.shape[1]:
        raise ValueError(f"mode_index={mode_index} out of range - only {coeff.shape[1]} mode(s) available.")

    a_k = coeff[:, mode_index]
    n_frames = len(a_k)
    t = np.arange(n_frames) * dt if dt is not None else np.arange(n_frames)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    ax.plot(t, a_k, color='black', linewidth=1.2)
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.4, linestyle='--')
    ax.set_xlabel('Time [s]' if dt is not None else 'Frame index')
    ax.set_ylabel(f'$a_{{{mode_index + 1}}}(t)$')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax


def plot_pod_spectrogram(pod_result: dict, dt: float = None, fs: float = None, n_modes: int = None,
                          n_chunk: int = 4, nperseg: int = None, nfft: int = None, detrend='constant',
                          cmap: str = 'cividis', ax=None, savepath: str = None, dpi: int = 600, **welch_kwargs):

    '''
    Welch PSD of EVERY retained POD mode's own temporal coefficient
    a_k(t) (not just one, as plot_pod_temporal_coefficient()/
    bladeprocessor.convergence.plot_periodogram() examine individually),
    contoured together as mode index vs. frequency - generalizes the
    single-mode PSD check into one figure showing how a dataset's
    characteristic frequencies are distributed across its whole modal
    decomposition, not just whichever one or two modes were picked out
    for a closer look.

    Each mode's own PSD row is normalized by ITS OWN maximum before
    plotting (not by the global maximum across all modes) - otherwise a
    handful of low-order, high-ENERGY modes (see plot_pod_energy()) would
    wash out every higher-order mode's own spectral content under a
    shared color scale, even though a mode's FREQUENCY content is a
    separate question from how much of the total fluctuation ENERGY it
    carries (see rotaris-docs/pod_section.tex's own
    "Relation to the Harmonic Decomposition" subsection: POD ranks by
    energy, with no frequency assumption built in - this plot is exactly
    how that frequency content is then examined, across the whole
    decomposition at once). Per-row normalization trades away any
    information about one mode's PSD being larger in an absolute sense
    than another's - plot_pod_energy() already shows that - in exchange
    for making every mode's own dominant frequency equally visible.

    A single horizontal band of color (a given frequency lit up across
    many consecutive mode indices) indicates that one physical frequency
    is spread across several modes rather than isolated in one (POD does
    not guarantee a single frequency maps to a single mode - a traveling
    structure in particular typically needs a pair, see mode_index=0/1's
    own cross-check). A diagonal trend (characteristic frequency
    increasing with mode index) is the usual broadband-turbulence
    signature - higher-order, lower-energy modes generally carrying
    higher-frequency, more spatially fine-grained content.

    Parameters
    ----------
    pod_result : dict
        pod()'s own return value.
    dt, fs : see bladeprocessor.convergence.periodogram() - one of the
        two is REQUIRED.
    n_modes : int, optional
        How many modes (starting from the most energetic) to include -
        all modes present in `pod_result` by default. Keep this small
        deliberately for a first look (e.g. 10-20) - Welch's method run
        once per mode is not expensive, but a contour over hundreds of
        modes is mostly uninformative, low-energy noise past whatever
        point plot_pod_energy() already shows the spectrum flattening
        out.
    n_chunk, nperseg, nfft, detrend, **welch_kwargs : see
        bladeprocessor.convergence.periodogram() - identical meaning,
        applied identically to every mode (computed ONCE from the first
        mode's own frame count, not re-derived per mode, so every row
        shares exactly the same frequency axis and can be stacked into
        one contour).
    cmap : str
        Sequential colormap for the normalized PSD - 'cividis' by
        default, this project's own established colormap convention
        (see bladeprocessor.case_comparison._strip_colors()).

    Returns
    -------
    (fig, ax)
    '''

    coeff = pod_result['coefficients']
    n_frames, n_modes_total = coeff.shape
    n_modes = n_modes_total if n_modes is None else min(n_modes, n_modes_total)

    # Computed once here (mirroring periodogram()'s own n_chunk-based
    # default), then passed explicitly to every mode's own call below -
    # letting each call re-derive its own nperseg/nfft from `len(series)`
    # would still agree in practice (every mode has the same n_frames),
    # but doing it once and passing it explicitly is what actually
    # GUARANTEES every row shares one frequency axis, not just something
    # that happens to hold given today's input shapes.
    if nperseg is None:
        nperseg = n_frames // n_chunk
    if nfft is None:
        nfft = 1 << (nperseg - 1).bit_length()

    psd_rows = []
    for k in range(n_modes):
        freq, psd = periodogram(coeff[:, k], dt=dt, fs=fs, nperseg=nperseg, nfft=nfft,
                                 detrend=detrend, **welch_kwargs)
        psd_rows.append(psd)
    psd_grid = np.array(psd_rows)  # (n_modes, n_freq)

    # Skip freq=0 (DC) - undefined on a log x-axis, same convention as
    # bladeprocessor.convergence.plot_periodogram().
    start = 1 if freq[0] == 0 else 0
    freq = freq[start:]
    psd_grid = psd_grid[:, start:]

    row_max = psd_grid.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0  # an all-zero mode (e.g. a genuinely silent high-order mode) stays all-zero, not NaN
    psd_norm = psd_grid / row_max

    mode_idx = np.arange(1, n_modes + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(freq, mode_idx, psd_norm, cmap=cmap, shading='auto',
                          norm=LogNorm(vmin=max(psd_norm[psd_norm > 0].min(), 1e-6), vmax=1.0))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('POD mode')
    fig.colorbar(mesh, ax=ax, label='Normalized PSD [-]')
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax


def pod_mode_phase(pod_result: dict, mode_a: int = 0, mode_b: int = 1, dt: float = None, fs: float = None,
                    f_target: float = None, n_chunk: int = 4, nperseg: int = None, nfft: int = None,
                    detrend='constant', **csd_kwargs):

    '''
    Cross-spectral phase between two POD modes' own temporal
    coefficients, a_a(t) and a_b(t) - the quantitative version of the
    "do modes a and b form a traveling-wave pair" check already
    discussed qualitatively (comparable energy, see plot_pod_energy();
    comparable characteristic frequency, see plot_pod_spectrogram()). A
    genuinely traveling (as opposed to spatially stationary) structure
    requires two modes in QUADRATURE - a phase difference of
    approximately +/-90 degrees between their temporal coefficients,
    the direct 1D analogue of the classical cos(kx)/sin(kx) spatial
    quadrature pair needed to represent a traveling wave
    A*cos(kx - w*t) = A*[cos(kx)cos(wt) + sin(kx)sin(wt)]. A phase
    difference near 0 or +/-180 degrees instead indicates the two modes
    are in (or out of) phase with each other - describing a spatially
    STATIONARY pattern whose amplitude happens to be split across two
    modes (e.g. for a numerical reason, not a physical traveling-wave
    one), not a convecting one.

    Evaluated at whichever frequency the two coefficients are most
    coherently related (the frequency of peak magnitude-squared
    coherence between them, via scipy.signal.coherence), not at an
    arbitrary or pre-assumed frequency - a phase difference is only
    meaningful where the two signals are actually related; elsewhere it
    is, in effect, the phase difference of two unrelated noise
    processes. Pass `f_target` explicitly (e.g. the frequency already
    identified from plot_pod_spectrogram()/plot_periodogram()) to read
    the phase off at a specific, chosen frequency instead.

    Parameters
    ----------
    pod_result : dict
        pod()'s own return value.
    mode_a, mode_b : int
        0-indexed mode numbers to compare (0, 1 by default - the first
        and second, i.e. most energetic, modes).
    dt, fs : see bladeprocessor.convergence.periodogram() - one of the
        two is REQUIRED.
    f_target : float, optional
        Evaluate the phase at this specific frequency [Hz] instead of
        the frequency of peak coherence.
    n_chunk, nperseg, nfft, detrend, **csd_kwargs : see
        bladeprocessor.convergence.periodogram() - identical meaning,
        passed to both scipy.signal.csd and scipy.signal.coherence.

    Returns
    -------
    dict
        'frequency' : float [Hz] - the frequency the phase was read off
            at (either f_target, or the frequency of peak coherence).
        'phase_deg' : float, in (-180, 180] - the phase of a_b relative
            to a_a at that frequency; positive means a_b leads a_a.
        'coherence' : float, in [0, 1] - magnitude-squared coherence at
            that frequency; how reliable 'phase_deg' actually is (close
            to 1 is a clean, well-defined phase relationship; close to 0
            means the two signals are essentially unrelated there, and
            'phase_deg' should not be trusted).
        'deviation_from_quadrature_deg' : float, in [0, 90] - how far
            'phase_deg' sits from the nearest of +90/-90 degrees - 0
            means perfect quadrature (the traveling-wave signature);
            90 means perfectly in/out of phase (the stationary-pattern
            signature).
    '''

    a_a = pod_result['coefficients'][:, mode_a]
    a_b = pod_result['coefficients'][:, mode_b]
    n_frames = len(a_a)

    if fs is None:
        if dt is None:
            raise ValueError("pod_mode_phase() needs fs or dt - a frequency axis is meaningless without one.")
        fs = 1.0 / dt

    if nperseg is None:
        nperseg = n_frames // n_chunk
    if nfft is None:
        nfft = 1 << (nperseg - 1).bit_length()

    freq, Pxy = csd(a_a, a_b, fs=fs, nperseg=nperseg, nfft=nfft, detrend=detrend, **csd_kwargs)
    _, Cxy = coherence(a_a, a_b, fs=fs, nperseg=nperseg, nfft=nfft, detrend=detrend, **csd_kwargs)

    start = 1 if freq[0] == 0 else 0  # skip DC, same convention as periodogram()/plot_periodogram()
    freq, Pxy, Cxy = freq[start:], Pxy[start:], Cxy[start:]

    if f_target is None:
        idx = np.argmax(Cxy)
    else:
        idx = np.argmin(np.abs(freq - f_target))

    phase_deg = np.degrees(np.angle(Pxy[idx]))
    deviation_from_quadrature = abs(abs(phase_deg) - 90.0)

    return {
        'frequency': freq[idx],
        'phase_deg': phase_deg,
        'coherence': Cxy[idx],
        'deviation_from_quadrature_deg': deviation_from_quadrature,
    }
