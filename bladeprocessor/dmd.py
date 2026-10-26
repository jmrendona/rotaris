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


def _fit_dmd_operator(coeff, dt):

    '''
    The core DMD fit - k x k normal-equations least squares plus an
    eigendecomposition - factored out of dmd() so joint_dmd() (fit on a
    STACKED, multi-field coefficient array) can reuse the exact same,
    already-safe math instead of duplicating it. See dmd()'s own
    docstring for the full derivation; this only takes the already-
    assembled (n_frames, k) coefficient array and a timestep, with no
    assumption about where `coeff`'s own columns came from (a single
    field's POD coefficients, or several fields' stacked together).

    Returns
    -------
    mu, frequency, growth_rate, b, W : each sorted by DESCENDING |b| -
        see dmd()'s own Returns section for what each one is. W's
        columns are the (unit-norm) eigenvectors, in the SAME coefficient
        basis/order `coeff`'s own columns were given in.
    '''

    n_frames, k = coeff.shape
    if n_frames < 3:
        raise ValueError(f"Need at least 3 frames to fit a one-step DMD operator - got {n_frames}.")

    Z1 = coeff[:-1, :].T  # (k, n_frames-1) - snapshots a(t_0) .. a(t_{N-2})
    Z2 = coeff[1:, :].T   # (k, n_frames-1) - the one-step-later snapshots a(t_1) .. a(t_{N-1})

    # Least-squares fit of Atilde (Z2 ~= Atilde @ Z1) via the k x k normal
    # equations - always safe regardless of n_frames, since the matrix
    # actually inverted here is only (k, k), never (n_frames, n_frames)
    # or (n_points, n_points).
    G1 = Z1 @ Z1.T
    Atilde = (Z2 @ Z1.T) @ np.linalg.pinv(G1)

    mu, W = np.linalg.eig(Atilde)  # W's columns are unit-norm eigenvectors (numpy's own eig() convention)

    omega = np.log(mu) / dt
    growth_rate = omega.real
    frequency = omega.imag / (2.0 * np.pi)

    b = np.linalg.solve(W, coeff[0, :])

    order = np.argsort(-np.abs(b))

    return mu[order], frequency[order], growth_rate[order], b[order], W[:, order]


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
    (descending |b_j|; see 'amplitude' below). The same b_j, mu_j (and
    'eigenvectors' below) also let the fitted dynamics be replayed
    forward in time, independent of rank - see dmd_reconstruct(), the
    standard way to check how well (or how quickly not) this fit
    actually describes the recorded signal.

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
        eigenvectors : np.ndarray, shape (k, k), complex
            W, columns = w_j, in POD-COEFFICIENT space (not yet
            projected through pod_result['modes']) - needed by
            dmd_reconstruct() to replay the fitted coefficient
            trajectory a_DMD(t) for comparison against
            pod_result['coefficients'] itself; not generally needed
            otherwise (use 'modes' for the physical-space shape).
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

    mu, frequency, growth_rate, b, W = _fit_dmd_operator(a, dt)
    modes = W.T @ psi

    return {
        'eigenvalues': mu,
        'frequency': frequency,
        'growth_rate': growth_rate,
        'amplitude': b,
        'modes': modes,
        'eigenvectors': W,
        'dt': dt,
    }


def dmd_reconstruct(dmd_result: dict, n_frames: int):

    '''
    Replays DMD's own fitted linear dynamics forward in time from its
    eigenvalues, amplitudes, and (POD-coefficient-space) eigenvectors
    alone, g(t_n)'s reduced-coordinate counterpart
    a_DMD(t_n) = sum_j b_j * mu_j^n * w_j, directly comparable to the
    ORIGINAL pod_result['coefficients'] dmd() was fit from (see
    plot_dmd_reconstruction()) - the standard way to check how well a
    DMD fit actually describes the recorded signal, not just read that
    off indirectly from its growth rates.

    The reconstruction is EXACT at n=0 by construction (that is exactly
    how the amplitudes b_j were derived - solving a(t_0) = W @ b) and
    then evolves purely from the fitted dynamics onward; this is honest
    behavior, not a bug - a reconstruction that matches at t=0 and then
    rapidly diverges from (or decays faster than) the real recorded
    signal is itself the direct, visual demonstration that a linear fit
    does not sustain the recorded behavior, exactly as a strongly
    negative growth_rate already indicates numerically. Conversely, a
    reconstruction that keeps tracking the real signal for a long
    stretch of the record is direct, visual evidence of a genuinely
    well-described, persistent oscillator.

    Parameters
    ----------
    dmd_result : dict
        dmd()'s own return value - needs 'eigenvalues', 'amplitude', and
        'eigenvectors' (NOT provided by joint_dmd(), which returns
        'modes_a'/'modes_b' instead of a single POD-coefficient space -
        this function is for a single-field dmd() result only).
    n_frames : int
        How many frames to replay forward - pass
        pod_result['coefficients'].shape[0] to reconstruct across the
        SAME record dmd() was fit from (the usual case, for a direct,
        frame-by-frame comparison).

    Returns
    -------
    coeff_recon : np.ndarray, shape (n_frames, k), complex
        a_DMD(t_n), one column per retained POD coefficient, in the SAME
        order as dmd_result['eigenvectors']'s own rows (i.e. the
        original POD mode order, NOT re-sorted by DMD amplitude - take
        .real for comparison/plotting; for a real input field, the
        imaginary part should already be negligible (machine precision)
        as a direct consequence of every oscillatory mode's complex-
        conjugate pairing (see dmd()'s own docstring) - a non-negligible
        imaginary part here would indicate something upstream (an
        unpaired mode, a bug) rather than real physics.
    '''

    mu = dmd_result['eigenvalues']
    b = dmd_result['amplitude']
    W = dmd_result['eigenvectors']

    powers = mu[None, :] ** np.arange(n_frames)[:, None]  # (n_frames, k) - mu_j^n
    coeff_recon = (powers * b[None, :]) @ W.T

    return coeff_recon


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


def plot_dmd_reconstruction(pod_result: dict, dmd_result: dict, mode_index: int = 0, dt: float = None,
                             ax=None, color: str = 'black', recon_color: str = 'tab:red',
                             savepath: str = None, dpi: int = 600):

    '''
    Overlays dmd_reconstruct()'s own replayed coefficient trajectory
    against the ACTUAL recorded POD coefficient a_ell(t) for one mode -
    the direct, visual DMD-fit-quality check (see dmd_reconstruct()'s
    own docstring): agreement over a long stretch of the record means
    the fitted dynamics genuinely describe the signal; the reconstruction
    peeling away from the real trace (typically decaying toward zero,
    given every growth rate found so far in this project's own data has
    been negative) is the same "not a persistent linear oscillator"
    conclusion already read off growth_rate, now shown directly rather
    than inferred from a single number.

    mode_index here indexes pod_result['coefficients']'s OWN mode order
    (0 = POD's most energetic mode) - NOT dmd_result's own DMD-amplitude
    ranking (dmd_reconstruct() returns every retained coefficient
    reconstructed at once, in POD's original order, by design, so this
    plot can be made for whichever POD mode is of interest).

    Returns
    -------
    (fig, ax)
    '''

    coeff = pod_result['coefficients']
    n_frames = coeff.shape[0]
    if mode_index >= coeff.shape[1]:
        raise ValueError(f"mode_index={mode_index} out of range - only {coeff.shape[1]} mode(s) available.")

    coeff_recon = dmd_reconstruct(dmd_result, n_frames)

    t = np.arange(n_frames) * dt if dt is not None else np.arange(n_frames)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    ax.plot(t, coeff[:, mode_index], color=color, linewidth=1.2, label='Recorded $a(t)$')
    ax.plot(t, coeff_recon[:, mode_index].real, color=recon_color, linewidth=1.2,
            linestyle='--', label='DMD reconstruction')
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.4, linestyle=':')
    ax.set_xlabel('Time [s]' if dt is not None else 'Frame index')
    ax.set_ylabel(f'$a_{{{mode_index + 1}}}(t)$')
    ax.grid(True, alpha=0.3)
    ax.legend()

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


def resample_coefficients(coeff, dt_from, t_target):

    '''
    Linear-interpolates a POD coefficient trajectory, recorded on its
    own native, regularly-spaced time grid (spacing dt_from), onto an
    externally given target time grid `t_target` - the resampling step
    joint_dmd() needs before two fields recorded at DIFFERENT native
    sampling rates (e.g. this project's own dt_forces != dt_pressure)
    can be stacked into one shared state and fit with a single DMD
    operator, which only makes sense once both share a common, single
    notion of "one step forward". Same plain np.interp-based approach
    already used by bladeprocessor.convergence.cycle_correlation() to
    resample onto a shared phase grid - standalone here rather than
    imported from there, since that function's own resampling is
    specific to one revolution's worth of phase, not an arbitrary
    external time grid.

    Interpolating is only ever safe DOWN in rate - t_target should be a
    COARSER (larger-spacing) grid than dt_from's own native one (see
    joint_dmd(), which enforces this); interpolating a signal onto a
    FINER grid than it was actually sampled at cannot recover content
    that was never recorded, it only manufactures smooth-looking values
    between real samples.

    Parameters
    ----------
    coeff : array-like, shape (n_frames, k)
        A POD coefficient trajectory on its own native, regular grid
        t_native = arange(n_frames) * dt_from.
    dt_from : float
        `coeff`'s own native timestep [s].
    t_target : array-like, shape (n_target,)
        The time values [s] to resample onto - must lie within
        `coeff`'s own recorded time span (t_native[0] to t_native[-1]);
        values outside that range are not extrapolated meaningfully by
        np.interp (it holds the nearest endpoint constant instead), so
        joint_dmd() restricts t_target to the actual overlap of both
        records before calling this.

    Returns
    -------
    coeff_resampled : np.ndarray, shape (n_target, k)
    '''

    coeff = np.asarray(coeff)
    t_target = np.asarray(t_target)
    n_frames = coeff.shape[0]
    t_native = np.arange(n_frames) * dt_from

    coeff_resampled = np.empty((len(t_target), coeff.shape[1]), dtype=coeff.dtype)
    for j in range(coeff.shape[1]):
        coeff_resampled[:, j] = np.interp(t_target, t_native, coeff[:, j])

    return coeff_resampled


def joint_dmd(pod_result_a: dict, dt_a: float, pod_result_b: dict, dt_b: float,
              n_modes_a: int = None, n_modes_b: int = None):

    '''
    Multi-variable ("coupled") DMD: fits ONE shared linear operator
    across the STACKED POD coefficients of two DIFFERENT fields at once
    (e.g. this project's own Cf and Cp) - a fundamentally different
    question from calling dmd() separately on each field and comparing
    the two fits side by side (as skin_friction_manager.py/
    pressure_manager.py already do). Two separate fits can only ever
    show that both fields happen to share a similar frequency; a joint
    fit instead directly tests whether a SINGLE, COMBINED linear
    operator meaningfully mixes both fields' own dynamics together -
    i.e. whether their fluctuations are coupled, not merely coincident.

    Mechanically: let a(t) be field A's own first n_modes_a retained POD
    coefficients and c(t) field B's own first n_modes_b, each a plain
    pod()['coefficients'] array. The combined state is their
    concatenation at each shared time instant, [a(t), c(t)], and
    _fit_dmd_operator() (the SAME normal-equations fit dmd() itself
    uses, see that function's own docstring) is applied to this single,
    wider coefficient array - A and B are not treated specially by the
    fit itself, only by how the result is unpacked afterward (see
    Returns below).

    Resampling onto a common time grid: "one step forward" is only
    well-defined once both fields share a single timeline, which they
    generally do NOT at this project's own native sampling (dt_forces
    != dt_pressure). Whichever field has the SMALLER dt (the finer,
    higher-frequency sampling) is linearly interpolated DOWN
    (resample_coefficients()) onto the OTHER field's own, coarser native
    grid, truncated to the two records' overlapping duration - never the
    other direction, since interpolating a coarse signal onto a finer
    grid cannot recover content it was never sampled at in the first
    place. This necessarily discards whatever content in the finer-
    sampled field sits above the COARSER field's own Nyquist frequency
    (half its sampling rate) - if that field's own plain dmd()/
    periodogram already identified a characteristic frequency above that
    limit, treat any result here involving that field with proportionally
    more caution; which field paid this cost is reported explicitly (see
    'resampled_field').

    Reading 'fraction_a'/'fraction_b': unlike dmd()'s own single-field
    modes (which automatically carry unit norm - see that function's own
    docstring), a joint mode's eigenvector splits across the A-block and
    B-block of the combined state, and the two pieces' own squared norms
    (which sum to exactly 1, since the full eigenvector is unit-norm) say
    how much of that particular mode's dynamics live in each field.
    fraction_a (or fraction_b) near 1 means that mode is essentially
    A-only (or B-only) dynamics incidentally present in the joint fit,
    not evidence of coupling; fraction_a/fraction_b both near 0.5 is the
    signature actually worth reporting - a mode whose fitted dynamics
    genuinely mix both fields together.

    Parameters
    ----------
    pod_result_a, pod_result_b : dict
        bladeprocessor.pod.pod()'s own return value, one per field.
    dt_a, dt_b : float
        Each field's own native timestep [s] (e.g. cfg.dt_forces,
        cfg.dt_pressure) - REQUIRED, both, since resampling needs a real
        physical time axis for each field individually, unlike dmd()'s
        own fs/dt alternative.
    n_modes_a, n_modes_b : int, optional
        How many of each field's own POD modes to include - all modes
        present in that field's own pod_result by default; see dmd()'s
        own `n_modes` docstring for why keeping these small is a
        deliberate noise-mitigation choice, now doubly so with two
        fields' worth of modes combined into one fit.

    Returns
    -------
    dict with keys, each sorted by DESCENDING |amplitude|:
        eigenvalues, frequency, growth_rate, amplitude : see dmd()'s own
            Returns section - now describing the JOINT fit.
        modes_a : np.ndarray, shape (k, n_points_a), complex
            Each joint mode's own spatial pattern projected through
            field A's POD modes alone (pod_result_a['modes']) - pass a
            dict like {'modes': result['modes_a'], 'frequency':
            result['frequency'], 'growth_rate': result['growth_rate']}
            to plot_dmd_mode() directly to visualize one, exactly as for
            a plain dmd() result.
        modes_b : np.ndarray, shape (k, n_points_b), complex
            Same, for field B.
        fraction_a, fraction_b : np.ndarray, shape (k,), real, in [0, 1]
            fraction_a + fraction_b == 1 for every mode - see above.
        eigenvectors : np.ndarray, shape (k_a + k_b, k), complex
            The full, stacked eigenvectors (rows 0:k_a are field A's own
            block, rows k_a: are field B's) - not generally needed
            directly; 'modes_a'/'modes_b' are already the physical-space
            projection most uses want.
        dt : float [s]
            The common (coarser) timestep the joint fit was actually
            computed at.
        resampled_field : 'a' or 'b'
            Which field was interpolated down onto the other's grid.
    '''

    k_a_total = pod_result_a['coefficients'].shape[1]
    k_b_total = pod_result_b['coefficients'].shape[1]
    k_a = k_a_total if n_modes_a is None else min(n_modes_a, k_a_total)
    k_b = k_b_total if n_modes_b is None else min(n_modes_b, k_b_total)

    coeff_a = pod_result_a['coefficients'][:, :k_a]
    coeff_b = pod_result_b['coefficients'][:, :k_b]

    # Whichever field has the LARGER dt (coarser/lower-frequency
    # sampling) sets the common grid; the OTHER (finer) field is
    # resampled down onto it, truncated to their overlapping duration -
    # see this function's own docstring for why this direction only.
    if dt_a >= dt_b:
        dt_common = dt_a
        t_common = np.arange(coeff_a.shape[0]) * dt_common
        t_overlap_end = (coeff_b.shape[0] - 1) * dt_b
        t_common = t_common[t_common <= t_overlap_end]
        coeff_a_use = coeff_a[:len(t_common), :]
        coeff_b_use = resample_coefficients(coeff_b, dt_b, t_common)
        resampled_field = 'b'
    else:
        dt_common = dt_b
        t_common = np.arange(coeff_b.shape[0]) * dt_common
        t_overlap_end = (coeff_a.shape[0] - 1) * dt_a
        t_common = t_common[t_common <= t_overlap_end]
        coeff_b_use = coeff_b[:len(t_common), :]
        coeff_a_use = resample_coefficients(coeff_a, dt_a, t_common)
        resampled_field = 'a'

    if len(t_common) < 3:
        raise ValueError(
            f"Only {len(t_common)} overlapping, common-grid frame(s) between the two records at "
            f"dt_common={dt_common} - need at least 3 to fit a one-step DMD operator. Check dt_a/dt_b "
            "and each field's own recorded duration."
        )

    combined = np.concatenate([coeff_a_use, coeff_b_use], axis=1)  # (n_common, k_a + k_b)

    mu, frequency, growth_rate, b, W = _fit_dmd_operator(combined, dt_common)

    W_a = W[:k_a, :]
    W_b = W[k_a:, :]
    norm_a_sq = np.sum(np.abs(W_a) ** 2, axis=0)
    norm_b_sq = np.sum(np.abs(W_b) ** 2, axis=0)
    total = norm_a_sq + norm_b_sq
    fraction_a = norm_a_sq / total
    fraction_b = norm_b_sq / total

    psi_a = pod_result_a['modes'][:k_a, :]
    psi_b = pod_result_b['modes'][:k_b, :]
    modes_a = W_a.T @ psi_a
    modes_b = W_b.T @ psi_b

    return {
        'eigenvalues': mu,
        'frequency': frequency,
        'growth_rate': growth_rate,
        'amplitude': b,
        'modes_a': modes_a,
        'modes_b': modes_b,
        'fraction_a': fraction_a,
        'fraction_b': fraction_b,
        'eigenvectors': W,
        'dt': dt_common,
        'resampled_field': resampled_field,
    }


def plot_joint_dmd_spectrum(joint_result: dict, ax=None, cmap: str = 'coolwarm',
                             savepath: str = None, dpi: int = 600):

    '''
    joint_dmd()'s own version of plot_dmd_spectrum() - frequency vs.
    growth rate, marker SIZE set by |amplitude| as before, but marker
    COLOR now set by fraction_a instead of amplitude, on a DIVERGING
    colormap centered at 0.5 - 0 (one end) is a B-only mode, 1 (other
    end) an A-only mode, and the pale/neutral midpoint colors are the
    modes actually worth reporting: dynamics the fit found genuinely
    mixing both fields together, not just the two fields' independent
    content coexisting in one combined matrix.

    Returns
    -------
    (fig, ax)
    '''

    freq = joint_result['frequency']
    growth = joint_result['growth_rate']
    amp = np.abs(joint_result['amplitude'])
    amp_max = amp.max() if amp.max() > 0 else 1.0
    size = np.clip(300 * amp / amp_max, 15, None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    sc = ax.scatter(freq, growth, s=size, c=joint_result['fraction_a'], cmap=cmap,
                     vmin=0, vmax=1, edgecolors='k', linewidths=0.5)
    ax.axhline(0, color='k', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Growth rate [1/s]')
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label='Fraction in field A [-] (0 = all B, 1 = all A, 0.5 = shared)')
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax
