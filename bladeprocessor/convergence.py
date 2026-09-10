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


def cumulative_mean(series, dt: float = None, rpm: float = None):

    '''
    Cumulative (running) mean of a scalar time series as a function of
    how many samples have been included so far - the standard "running
    mean vs. sample count" statistical-convergence check: a converged
    quantity's cumulative mean flattens to a horizontal asymptote as more
    data is included; a still-rising or still-oscillating one means more
    time/more revolutions are needed. See README.md's "Convergence
    checking" section for the general idea (tied to a signal's own
    integral/correlation time scale setting how many independent samples
    a given run actually contains - Pope, S.B., "Turbulent Flows",
    Cambridge University Press, 2000, discusses this general concept;
    no specific page is cited here - verify directly against the book's
    own index before citing a page number, rather than trust one from
    this codebase).

    Deliberately NOT tied to any one class - takes a plain 1D array, so
    it works equally on StripForces.total_loads()['thrust'] (or
    'torque'/'radial_force'/'tangential_force'), the spatial mean of
    FrictionLines.cf_time_series() (see plot_cf_phase_portrait()'s own
    spatial-mean approach for a ready-made way to collapse a Cf field to
    one scalar per frame first), or any other per-frame scalar series.

    Parameters
    ----------
    series : array-like, shape (n_frames,)
        Any scalar time series, one value per frame.
    dt : float, optional
        Physical timestep [s] between frames - if given, the x-axis is
        real elapsed time [s] instead of a bare frame index.
    rpm : float, optional
        Rotor speed [rev/min] - if given TOGETHER with dt, revolutions
        elapsed is also returned (None otherwise, since a frame count
        alone can't be converted to revolutions without a real dt).

    Returns
    -------
    x : np.ndarray, shape (n_frames,)
        Frame index (if dt is None), else elapsed time [s].
    running_mean : np.ndarray, shape (n_frames,)
        Cumulative mean of `series` up to and including each frame -
        running_mean[-1] equals series.mean() exactly.
    revolutions : np.ndarray, shape (n_frames,), or None
        Revolutions elapsed at each frame - only computed if both dt and
        rpm are given.
    '''

    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"series must be 1D (one value per frame), got shape {series.shape}")
    if len(series) < 2:
        raise ValueError(f"Need at least 2 frames for a running mean - got {len(series)}.")

    running_mean = np.cumsum(series) / np.arange(1, len(series) + 1)
    x = np.arange(len(series)) * dt if dt is not None else np.arange(len(series))

    revolutions = None
    if dt is not None and rpm is not None:
        revolutions = x * rpm / 60.0

    return x, running_mean, revolutions


def plot_cumulative_mean(series, dt: float = None, rpm: float = None, ylabel: str = None,
                          ax=None, color: str = 'tab:blue', savepath: str = None, dpi: int = 600):

    '''
    Plot cumulative_mean()'s running mean vs. elapsed time (or frame
    index, if dt isn't given) - with a second x-axis in revolutions along
    the top if rpm is ALSO given, so "how many cycles until this
    flattens out" is readable directly rather than having to convert
    frame counts by hand.

    Parameters
    ----------
    series, dt, rpm : see cumulative_mean().
    ylabel : str, optional
        Defaults to a generic 'Cumulative mean' - pass the actual
        quantity's own label (e.g. 'Thrust [N]', matching
        StripForces._COMPONENT_LABELS' style) for a properly labeled
        axis.

    Returns
    -------
    (fig, ax)
    '''

    x, running_mean, revolutions = cumulative_mean(series, dt=dt, rpm=rpm)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    ax.plot(x, running_mean, color=color, linewidth=1.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel('Time [s]' if dt is not None else 'Frame index')
    ax.set_ylabel(ylabel or 'Cumulative mean')
    ax.grid(True, alpha=0.4)

    if revolutions is not None:
        ax_top = ax.twiny()
        ax_top.set_xlim(revolutions[0], revolutions[-1])
        ax_top.set_xlabel('Revolutions included')

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax


def cumulative_stats(series, dt: float = None, rpm: float = None, sync_to_revolution: bool = False):

    '''
    Cumulative (running) MEAN and VARIANCE of a scalar time series as a
    function of how many samples have been included so far - mirrors
    Pope, "Turbulent Flows" (2000)'s own illustration of <U> and <u'^2>
    both approaching an asymptote as more samples/time are included, for
    a statistically stationary process (no specific page cited - verify
    against the book's own index).

    sync_to_revolution : bool
        If True (needs BOTH dt and rpm), the running stats are only
        evaluated at frames closest to a COMPLETE number of revolutions
        (1, 2, 3, ... full turns), not every single frame. This matters
        for a periodic-plus-turbulent signal (e.g. Cf/thrust on this
        project's own case, which has a real once-per-revolution
        component - see HANDOFF.md): a plain per-frame running mean of
        such a signal does NOT cleanly asymptote even once the run has
        genuinely converged - it shows a persistent, phase-dependent
        RIPPLE, since stopping partway through a revolution includes an
        unequal, arbitrary slice of the periodic cycle each time.
        Pope's own asymptotic picture implicitly assumes each new sample
        is a comparable statistical realization; for a periodic signal
        that's one whole revolution, not one arbitrary frame. This
        option evaluates the running stats only at those comparable
        points, which is usually what actually needs checking - if you
        DON'T see an asymptote even with this on, that's now much more
        likely a genuine "not converged yet" finding than a plotting
        artifact.

    Parameters
    ----------
    series, dt, rpm : see cumulative_mean().

    Returns
    -------
    x : np.ndarray
        Frame index or elapsed time [s] (see cumulative_mean()) -
        subsampled to revolution boundaries if sync_to_revolution=True.
    running_mean, running_var : np.ndarray, same shape as x
        running_var[-1] equals series.var() (population/biased
        variance - E[X^2] - E[X]^2 - the fluctuation-about-the-mean
        convention already used elsewhere in this project, e.g.
        FrictionLines.cf(stat='rms')) exactly, same guarantee as
        cumulative_mean()'s running_mean[-1] == series.mean().
    revolutions : np.ndarray, or None - see cumulative_mean().
    '''

    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"series must be 1D (one value per frame), got shape {series.shape}")
    if len(series) < 2:
        raise ValueError(f"Need at least 2 frames for running stats - got {len(series)}.")

    n = np.arange(1, len(series) + 1)
    running_mean = np.cumsum(series) / n
    running_meansq = np.cumsum(series ** 2) / n
    running_var = running_meansq - running_mean ** 2

    x = np.arange(len(series)) * dt if dt is not None else np.arange(len(series))
    revolutions = x * rpm / 60.0 if (dt is not None and rpm is not None) else None

    if sync_to_revolution:
        if dt is None or rpm is None:
            raise ValueError("sync_to_revolution=True needs BOTH dt and rpm.")
        frames_per_rev = 60.0 / (rpm * dt)
        n_revs = int(len(series) / frames_per_rev)
        if n_revs < 1:
            raise ValueError(
                f"This series ({len(series)} frames) doesn't even span one full revolution "
                f"({frames_per_rev:.1f} frames/rev at rpm={rpm}, dt={dt}) - can't sync to revolutions."
            )
        idx = np.clip(np.round(np.arange(1, n_revs + 1) * frames_per_rev).astype(int) - 1, 0, len(series) - 1)
        x, running_mean, running_var = x[idx], running_mean[idx], running_var[idx]
        revolutions = revolutions[idx]

    return x, running_mean, running_var, revolutions


def plot_cumulative_stats(series, dt: float = None, rpm: float = None, sync_to_revolution: bool = False,
                           ylabel: str = None, color: str = 'tab:blue', savepath: str = None, dpi: int = 600):

    '''
    Two-panel plot of cumulative_stats()'s running mean (top) and running
    variance (bottom) vs. elapsed time/revolutions - directly mirrors
    Pope's own <U>-and-<u'^2>-vs-sample-count figure (see that function's
    docstring for the citation caveat and what sync_to_revolution fixes
    for a periodic-plus-turbulent signal).

    Returns
    -------
    (fig, (ax_mean, ax_var))
    '''

    x, running_mean, running_var, revolutions = cumulative_stats(
        series, dt=dt, rpm=rpm, sync_to_revolution=sync_to_revolution)

    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    marker = 'o-' if sync_to_revolution else '-'
    ax_mean.plot(x, running_mean, marker, color=color, linewidth=1.5, markersize=4)
    ax_mean.set_ylabel(ylabel or r'$\langle X \rangle$')
    ax_mean.grid(True, alpha=0.4)

    ax_var.plot(x, running_var, marker, color=color, linewidth=1.5, markersize=4)
    ax_var.set_xlabel('Time [s]' if dt is not None else 'Frame index')
    ax_var.set_ylabel(r"$\langle x'^2 \rangle$")
    ax_var.grid(True, alpha=0.4)
    ax_var.set_xlim(x[0], x[-1])

    if revolutions is not None:
        ax_top = ax_mean.twiny()
        ax_top.set_xlim(revolutions[0], revolutions[-1])
        ax_top.set_xlabel('Revolutions included')

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, (ax_mean, ax_var)


def autocorrelation(series, max_lag: int = None):

    '''
    Sample autocorrelation function rho(s) = R(s)/R(0) of a scalar time
    series, for non-negative lags s = 0, 1, ..., max_lag - the standard
    single-realization (biased) time-average estimator. Pope, "Turbulent
    Flows" (2000), notes that for a TRUE, statistically stationary
    process, the (ensemble-averaged) autocovariance/autocorrelation are
    EVEN functions of the lag (no specific page cited - verify directly).

    IMPORTANT - and the reason this function only returns non-negative
    lags at all: the standard single-window estimator computed here is
    ALGEBRAICALLY GUARANTEED to be even (a direct consequence of
    correlating any finite sequence with a shifted copy of itself being
    exactly symmetric under relabeling - true for ANY finite sequence,
    stationary or not, not something that depends on the real data
    actually being stationary). Concretely: defining
    R_hat(-s) using the LAST (N-s) points as the reference is, term for
    term, identical to R_hat(+s) computed using the FIRST (N-s) points -
    a pure re-indexing identity. So computing this from ONE window and
    checking whether it "looks even" has NO diagnostic power - it will
    always pass trivially, regardless of whether the simulation has
    converged. What IS meaningful: comparing rho(s) computed from
    DIFFERENT, independent windows of the same run - see
    plot_autocorrelation_windows(), which is the actual convergence
    check this function is meant to feed.

    Parameters
    ----------
    series : array-like, shape (n,)
    max_lag : int, optional
        Largest lag to compute - defaults to n // 4 (a reliable estimate
        needs many more samples than the lag itself; using the full
        range up to n-1 gets extremely noisy - fewer and fewer pairs of
        points contribute - at large lags).

    Returns
    -------
    lags : np.ndarray, shape (max_lag + 1,) - 0, 1, ..., max_lag
    rho : np.ndarray, shape (max_lag + 1,) - rho[0] == 1.0 always.
    '''

    series = np.asarray(series, dtype=float)
    n = len(series)
    if max_lag is None:
        max_lag = max(n // 4, 1)
    if max_lag >= n:
        raise ValueError(f"max_lag ({max_lag}) must be less than the series length ({n}).")

    x = series - series.mean()
    var = np.mean(x ** 2)
    if var == 0:
        raise ValueError("series has zero variance - autocorrelation is undefined.")

    rho = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        rho[lag] = 1.0 if lag == 0 else np.mean(x[:-lag] * x[lag:]) / var

    return np.arange(max_lag + 1), rho


def plot_autocorrelation_windows(series, n_windows: int = 2, max_lag: int = None, dt: float = None,
                                  labels=None, ax=None, cmap: str = 'cividis',
                                  savepath: str = None, dpi: int = 600):

    '''
    The MEANINGFUL version of the "R(s) is even for a stationary process"
    check from Pope (see autocorrelation()'s docstring for why checking
    a single window's own symmetry has no power to detect non-
    convergence - it's guaranteed by construction either way). Stationary
    means the statistics are TIME-INVARIANT, not just that one estimate
    looks symmetric - so this splits the series into `n_windows`
    sequential, non-overlapping blocks, computes each block's OWN
    autocorrelation function independently, and overlays them.

    If the run has genuinely reached a statistically stationary state,
    these independent estimates should closely agree with each other
    (an early window's correlation structure looks like a late window's).
    If the process is still evolving, they won't - e.g. a still-transient
    run might show faster/slower decay or a level offset between windows,
    and a signal with an unremoved periodic component will show an
    undamped oscillation that never decays to zero within the lag range
    shown (correctly reflecting that a periodic signal is not, in the
    classical random-process sense, "decorrelating" at all - see
    StripForces.phase_lock() to fold that component out first if you want
    to look at the residual turbulence's own correlation structure
    specifically, instead).

    Parameters
    ----------
    series : array-like, shape (n,)
    n_windows : int
        How many sequential, non-overlapping blocks to split `series`
        into and compare - 2 (first half vs. second half) is the
        simplest, most standard choice; more windows give a finer-
        grained (but noisier per-window) view of when things stabilized.
    max_lag : int, optional
        Passed to autocorrelation() for EACH window - since each window
        has n/n_windows samples, not n, the default (that window's own
        length // 4) is smaller than autocorrelation()'s default would
        be on the full series.
    dt : float, optional
        If given, the lag axis is in seconds instead of frames.
    labels : list of str, optional
        One label per window - defaults to 'Window 1', 'Window 2', ...

    Returns
    -------
    (fig, ax)
    '''

    series = np.asarray(series, dtype=float)
    n = len(series)
    block = n // n_windows
    if block < 8:
        raise ValueError(
            f"Only {block} samples per window with n_windows={n_windows} (series length {n}) - "
            "too few for a meaningful autocorrelation; use fewer windows or a longer series."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_windows))

    for i, color in enumerate(colors):
        start = i * block
        end = (i + 1) * block if i < n_windows - 1 else n
        lags, rho = autocorrelation(series[start:end], max_lag=max_lag)
        x_lag = lags * dt if dt is not None else lags
        label = labels[i] if labels is not None else f'Window {i + 1}'
        ax.plot(x_lag, rho, color=color, label=label)

    ax.axhline(0, color='k', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Lag [s]' if dt is not None else 'Lag [frames]')
    ax.set_ylabel(r'$\rho(s)$ [-]')
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax
