'''
Dynamic Mode Decomposition (DMD), projected onto an already-computed
POD basis (bladeprocessor.pod.pod()'s own return value) rather than
fit directly from the raw (n_frames, n_points) surface field.

Why projected, not "exact" DMD on the raw field: standard/exact DMD
(Tu et al., 2014) needs the economy SVD of the (n_points, n_frames-1)
snapshot matrix - the exact same large, dense SVD that pod()'s own
docstring documents as confirmed to SEGFAULT (non-deterministically,
inside this cluster's LAPACK) at this project's real data scale.
Fitting DMD's linear operator instead to pod()'s own temporal
coefficients a_l(t) - already a safe, small (n_frames, k) array,
obtained via the method-of-snapshots trick pod() already implements -
sidesteps that large SVD entirely, at the cost of restricting the fit
to whatever subspace pod_result's own modes already span. This
restriction is not purely a workaround, either: a noise-robust DMD fit
generally NEEDS a prior rank truncation anyway (textbook/exact DMD is
well known to be considerably more sensitive to broadband noise than
POD itself - a real concern here, since this project's own Cf/Cp POD
energy spectra are not sharply dominated by one mode, i.e. a
non-negligible broadband component is already known to be present).
Truncating to pod_result's own first `n_modes` (see dmd()'s own
parameter) is exactly the standard mitigation - fit the dynamics only
in the subspace already identified as energetically significant, not
in whatever noisy, high-order directions a handful of samples cannot
meaningfully constrain anyway.

Where POD (bladeprocessor/pod.py) and DMD complement each other: POD
ranks spatial patterns by ENERGY with no assumption about their
frequency content (a mode's own a_l(t) must be examined separately,
e.g. via plot_pod_spectrogram()/pod_mode_phase(), to learn anything
about frequency at all). DMD instead fits a best-approximating LINEAR
operator to the recorded dynamics directly, and in exchange gets, for
every retained mode, a well-defined complex frequency AND growth/decay
rate in one step - not just where energy concentrates in frequency, but
whether that frequency content is itself growing, decaying, or held in
a statistically steady (neutral) oscillation. This is the one piece of
information Welch's method/plot_pod_spectrogram() cannot give at all
(a PSD has no concept of growth/decay, only of relative power), and is
exactly the missing piece needed to decide whether an identified
coherent structure (e.g. the LSB-related Cf/Cp POD mode pair already
discussed) is a genuinely sustained, repeatable oscillation (growth
rate near zero) or a transient feature that happens to still be present
somewhere in this particular recorded window (growth or decay rate
clearly nonzero).
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


def dmd(pod_result: dict, dt: float = None, fs: float = None, n_modes: int = None):

    '''
    Dynamic Mode Decomposition, fit in the reduced coordinates of an
    already-computed POD result (bladeprocessor.pod.pod()'s own
    'coefficients') - see this module's own docstring for why this
    "projected DMD" approach, rather than exact DMD on the raw field, is
    used here.

    Algorithm: let a(t_n) = pod_result['coefficients'][n, :k] (a
    k-dimensional vector, k = n_modes or fewer). DMD seeks the single
    k x k linear operator Atilde best explaining the recorded one-step
    advance, a(t_{n+1}) ~= Atilde @ a(t_n), fit here by ordinary least
    squares via the (tiny, always-safe) k x k normal equations - no SVD
    of any (n_frames, *) or (*, n_points)-sized matrix is involved at
    any point. Atilde's eigendecomposition, Atilde @ w_j = mu_j * w_j,
    gives k discrete-time eigenvalues mu_j; converted to continuous time
    via omega_j = ln(mu_j) / dt, their real part is each mode's own
    growth/decay rate and their imaginary part (divided by 2*pi) its
    frequency. Each eigenvector w_j maps back to a DMD mode in the
    ORIGINAL (surfel) space via Phi_j = w_j^T @ pod_result['modes'][:k] -
    a direct consequence of a(t)^T @ Psi (Psi = pod_result['modes'][:k])
    being the physical-space reconstruction at time t, with a(t)
    expressed in the Atilde eigenbasis. Conveniently, because
    pod_result's own spatial modes are already orthonormal
    (Psi @ Psi.T = I) and numpy's eig() normalizes every eigenvector to
    unit Euclidean norm, every Phi_j inherits EXACTLY unit 2-norm
    automatically (||w_j^T Psi||_2^2 = w_j^T Psi Psi^T w_j = w_j^T w_j =
    1) - so the initial-condition amplitude b_j (see below) is already a
    fair, directly comparable measure of each mode's own contribution,
    with no separate mode-norm correction needed.

    Mode amplitudes b_j are the coordinates of the FIRST recorded
    frame's own coefficient vector in the eigenvector basis,
    b = W^{-1} @ a(t_0) (W's columns are the w_j) - the standard "exact
    DMD" amplitude definition (Tu et al., 2014), used here to RANK modes
    (descending |b_j|; see 'amplitude' below) rather than to reconstruct
    the full field (no reconstruct_from_dmd()-style function is provided
    here - not currently needed, and straightforward to add from
    g(p, t_n) ~= sum_j b_j * mu_j^n * Phi_j(p) if it becomes useful).

    Reading the result (see plot_dmd_spectrum()):
        growth_rate ~= 0 : a genuinely SUSTAINED oscillation - the
            signature of a real, repeatable coherent structure (e.g. a
            limit-cycle-like vortex-shedding process), not a one-off
            transient specific to this particular recorded window.
        growth_rate < 0 : decaying - either a genuine physical transient
            still dying out in this record, or simply noise that this
            linear fit happens to approximate with a decaying complex
            exponential; low |amplitude| for such a mode supports the
            latter.
        growth_rate > 0 : growing - plausible early in a still-developing
            flow, but for an otherwise statistically stationary signal
            this is usually a fitting artifact (a linear operator fit to
            genuinely nonlinear turbulent dynamics can produce spurious
            unstable-looking eigenvalues) and should not be trusted
            without independent confirmation, e.g. by checking it
            reappears when the same fit is repeated on a different time
            window of the same record.
    Every mode of a REAL field appears either on its own (mu_j real,
    i.e. frequency exactly 0) or as part of a complex-conjugate pair
    (mu_j, mu_j^*) with matching growth rate, matching |amplitude|, and
    opposite-signed frequency - a single, unpaired nonzero-frequency
    mode is a numerical artifact, not a physical one.

    Parameters
    ----------
    pod_result : dict
        bladeprocessor.pod.pod()'s own return value.
    dt, fs : see bladeprocessor.convergence.periodogram() - one of the
        two is REQUIRED (growth rates/frequencies are meaningless
        without a real timestep).
    n_modes : int, optional
        How many of pod_result's own modes (starting from the most
        energetic) to fit the DMD operator on - all modes present in
        `pod_result` by default. See this module's own docstring for why
        keeping this SMALL (e.g. only as many modes as
        plot_pod_energy()'s own scree plot shows as clearly above the
        noise floor) is the standard, deliberate noise-mitigation choice
        here, not merely a performance optimization.

    Returns
    -------
    dict with keys, each sorted by DESCENDING |amplitude| (the most
    amplitude-significant mode first):
        eigenvalues : np.ndarray, shape (k,), complex
            Discrete-time eigenvalues mu_j of Atilde.
        frequency : np.ndarray, shape (k,) [Hz]
        growth_rate : np.ndarray, shape (k,) [1/s]
        amplitude : np.ndarray, shape (k,), complex
            b_j - see above; |amplitude| is the ranking criterion.
        modes : np.ndarray, shape (k, n_points), complex
            DMD spatial modes Phi_j, one per row (same row-per-mode
            convention as pod_result['modes']) - see plot_dmd_mode() to
            visualize one on the actual blade geometry.
        dt : float [s]
    '''

    a = pod_result['coefficients']
    n_frames, k_pod = a.shape
    k = k_pod if n_modes is None else min(n_modes, k_pod)
    a = a[:, :k]
    psi = pod_result['modes'][:k, :]

    if dt is None:
        if fs is None:
            raise ValueError("dmd() needs fs or dt - growth rates/frequencies are meaningless without one.")
        dt = 1.0 / fs
    if n_frames < 3:
        raise ValueError(f"Need at least 3 frames to fit a one-step DMD operator - got {n_frames}.")

    Z1 = a[:-1, :].T  # (k, n_frames-1) - snapshots a(t_0) .. a(t_{N-2})
    Z2 = a[1:, :].T   # (k, n_frames-1) - the one-step-later snapshots a(t_1) .. a(t_{N-1})

    # Least-squares fit of Atilde (Z2 ~= Atilde @ Z1) via the k x k normal
    # equations - always safe regardless of n_frames, since the matrix
    # actually inverted here is only (k, k) (k = a handful of retained
    # POD modes), never (n_frames, n_frames) or (n_points, n_points).
    G1 = Z1 @ Z1.T
    Atilde = (Z2 @ Z1.T) @ np.linalg.pinv(G1)

    mu, W = np.linalg.eig(Atilde)  # W's columns are unit-norm eigenvectors (numpy's own eig() convention)

    omega = np.log(mu) / dt
    growth_rate = omega.real
    frequency = omega.imag / (2.0 * np.pi)

    b = np.linalg.solve(W, a[0, :])
    modes = W.T @ psi

    order = np.argsort(-np.abs(b))

    return {
        'eigenvalues': mu[order],
        'frequency': frequency[order],
        'growth_rate': growth_rate[order],
        'amplitude': b[order],
        'modes': modes[order],
        'dt': dt,
    }


def plot_dmd_spectrum(dmd_result: dict, ax=None, cmap: str = 'cividis', savepath: str = None, dpi: int = 600):

    '''
    The standard DMD spectrum plot: every retained mode's frequency
    (x-axis) against its growth rate (y-axis), marker size/color both
    set by |amplitude| - the direct visual answer to "which modes are
    genuinely sustained oscillations" (growth rate near the dashed
    zero line), "which are decaying noise" (growth rate well below it,
    small/pale marker), and "is there a real, nonzero characteristic
    frequency at all, or only a near-zero-frequency, purely-growing-or-
    decaying mode" (see dmd()'s own docstring for the full reading, and
    the complex-conjugate-pair check for a REAL field).

    Returns
    -------
    (fig, ax)
    '''

    freq = dmd_result['frequency']
    growth = dmd_result['growth_rate']
    amp = np.abs(dmd_result['amplitude'])
    amp_max = amp.max() if amp.max() > 0 else 1.0
    size = np.clip(300 * amp / amp_max, 15, None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    sc = ax.scatter(freq, growth, s=size, c=amp, cmap=cmap, edgecolors='k', linewidths=0.5)
    ax.axhline(0, color='k', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Growth rate [1/s]')
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label='DMD amplitude $|b|$ [-]')
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax


def plot_dmd_mode(radius, chord, dmd_result: dict, mode_index: int = 0, part: str = 'real',
                   marker_size: float = 1, value_clip_percentile: float = 99, unit_label: str = None,
                   figsize: tuple = None, ax=None, savepath: str = None, dpi: int = 600,
                   radius_full=None, chord_full=None, background_color: str = '0.85'):

    '''
    ONE (necessarily complex) DMD spatial mode, plotted on the actual
    blade geometry - same raw-scatter style as
    bladeprocessor.pod.plot_pod_mode(), which this function otherwise
    mirrors exactly (including the radius_full/chord_full backdrop
    option - see that function's own docstring).

    part : 'real', 'imag', 'abs', or 'phase'
        A DMD mode Phi_j(p) is complex-valued even for a real physical
        field - its complex phase at each point encodes the structure's
        own CONVECTION direction (how the pattern's apparent spatial
        phase rotates from point to point), not merely a numerical
        artifact. 'real' (default) and 'imag' plot that component
        directly, symmetric about zero exactly like plot_pod_mode()'s
        own color scaling; 'abs' plots the mode's local magnitude
        (always >= 0, no sign to preserve); 'phase' plots its local
        phase angle in degrees, a cyclic quantity best read with a
        cyclic colormap (defaults to 'twilight').

    Returns
    -------
    (fig, ax)
    '''

    radius = np.asarray(radius)
    chord = np.asarray(chord)
    modes = dmd_result['modes']

    if mode_index >= modes.shape[0]:
        raise ValueError(f"mode_index={mode_index} out of range - only {modes.shape[0]} mode(s) available.")
    if radius.shape != chord.shape or radius.shape[0] != modes.shape[1]:
        raise ValueError(
            f"radius/chord (shape {radius.shape}) must match dmd_result['modes']'s own "
            f"n_points ({modes.shape[1]}) - same region selection used for both."
        )

    mode = modes[mode_index]

    if part == 'real':
        field, cmap, symmetric = mode.real, 'RdBu_r', True
    elif part == 'imag':
        field, cmap, symmetric = mode.imag, 'RdBu_r', True
    elif part == 'abs':
        field, cmap, symmetric = np.abs(mode), 'viridis', False
    elif part == 'phase':
        field, cmap, symmetric = np.degrees(np.angle(mode)), 'twilight', False
    else:
        raise ValueError(f"part must be 'real', 'imag', 'abs', or 'phase' - got '{part}'.")

    if (radius_full is None) != (chord_full is None):
        raise ValueError("radius_full and chord_full must be given together.")

    if ax is None:
        if figsize is None:
            figsize = (9, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if radius_full is not None:
        ax.scatter(np.asarray(radius_full), np.asarray(chord_full), s=marker_size,
                   color=background_color, linewidths=0, zorder=1)

    if symmetric:
        vmax = np.percentile(np.abs(field), value_clip_percentile) if field.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        field_clipped = np.clip(field, -vmax, vmax)
        vmin = -vmax
    elif part == 'phase':
        field_clipped = field
        vmin, vmax = -180.0, 180.0
    else:
        vmax = np.percentile(field, value_clip_percentile) if field.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        field_clipped = np.clip(field, 0, vmax)
        vmin = 0.0

    sc = ax.scatter(radius, chord, c=field_clipped, s=marker_size, cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

    freq = dmd_result['frequency'][mode_index]
    growth = dmd_result['growth_rate'][mode_index]
    ax.set_title(f'DMD mode {mode_index + 1} ({part}) - $f$={freq:.3g} Hz, growth={growth:.3g} 1/s')
    ax.set_xlabel('$r$ [m]')
    ax.set_ylabel('chord [m]')
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.18, aspect=30,
                 label=unit_label or f'DMD mode amplitude ({part}) [-]')

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax
