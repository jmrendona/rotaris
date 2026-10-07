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
hundreds of thousands of columns. UNLIKE FrictionLines/StripForces,
SurfaceVariable's own variable_time_series()/cp_time_series() do NOT
currently support any span_min/span_max cropping at the HDF5-read level
- they always read every point, every frame, of whichever Data/<surface>
variable is requested (see those methods' own docstrings). Column-
masking a SurfaceVariable-derived array AFTER calling
variable_time_series()/cp_time_series() (as shown in the examples below)
only reduces what pod() itself has to factor - it does not avoid the
eager full-array read happening first. For a large instantaneous
pressure file, this read could reproduce the same category of OOM this
project already fixed once for Cf/forces; if that turns out to be a
problem in practice, the fix is the same one already applied elsewhere -
adding masked, column-selective HDF5 reads to SurfaceVariable itself -
not yet done here since it wasn't asked for.

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
    cp_data = sv.cp_time_series(surface='Upper')[:, region]  # see memory note above
    result = pod(cp_data, n_modes=10)
    plot_pod_mode(radius[region], chord[region], result, mode_index=0,
                  unit_label='$C_p$ mode amplitude [-]',
                  savepath='pod_mode1_cp.png')
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


def pod(data, n_modes: int = None, subtract_mean: bool = True, weights=None):

    '''
    Proper Orthogonal Decomposition of a time-resolved scalar field via
    the singular value decomposition (SVD) - the "method of snapshots"
    approach, computed implicitly: since data is already (n_frames,
    n_points) with n_points typically far larger than n_frames for a
    CFD surface, numpy's economy SVD (full_matrices=False) on this exact
    shape is already the computationally efficient path (equivalent in
    cost to explicitly eigendecomposing the much smaller n_frames x
    n_frames Gram matrix, without needing to form that matrix by hand).

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
            before any n_modes truncation (the rank of the economy SVD).
    '''

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_frames, n_points), got shape {data.shape}")
    n_frames, n_points = data.shape
    if n_frames < 2:
        raise ValueError(f"Need at least 2 frames for a meaningful POD - got {n_frames}.")

    mean_field = data.mean(axis=0) if subtract_mean else None
    X = (data - mean_field[None, :]) if subtract_mean else data.copy()

    sqrt_w = None
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_points,):
            raise ValueError(f"weights must have shape ({n_points},), got {weights.shape}.")
        if np.any(weights < 0):
            raise ValueError("weights (surfel areas) must be non-negative.")
        sqrt_w = np.sqrt(weights)
        X = X * sqrt_w[None, :]

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n_modes_total = len(S)

    if sqrt_w is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            Vt = Vt / sqrt_w[None, :]
        Vt[:, weights == 0] = 0.0  # a zero-area/degenerate point carries no mode content

    k = n_modes_total if n_modes is None else min(n_modes, n_modes_total)

    energy_fraction_full = S ** 2 / np.sum(S ** 2)

    return {
        'modes': Vt[:k, :],
        'coefficients': U[:, :k] * S[None, :k],
        'singular_values': S[:k],
        'energy_fraction': energy_fraction_full[:k],
        'cumulative_energy': np.cumsum(energy_fraction_full)[:k],
        'mean': mean_field,
        'n_modes_total': n_modes_total,
    }


def plot_pod_mode(radius, chord, pod_result: dict, mode_index: int = 0,
                   cmap: str = 'RdBu_r', marker_size: float = 1, value_clip_percentile: float = 99,
                   unit_label: str = None, ax=None, savepath: str = None, dpi: int = 600):

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

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    sc = ax.scatter(radius, chord, c=mode_clipped, s=marker_size, cmap=cmap, vmin=-vmax, vmax=vmax)
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
