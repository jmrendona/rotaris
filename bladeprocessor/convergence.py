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


def _sync_indices(n_frames: int, dt: float = None, rpm: float = None,
                   sync: str = 'none', period_deg: float = None):

    '''
    Frame indices treated as one comparable "sample" for every running/
    windowed statistic in this module - defined ONCE here and shared by
    all of them, so "what counts as one sample" is always consistent
    across cumulative_mean()/cumulative_stats()/cumulative_moments()/
    integral_timescale()/standard_error(). Three modes, matching this
    project's actual range of cases:

    sync='none' (default): every single frame, 0..n_frames-1 - dt/rpm
        not required. Appropriate when each frame is already a
        reasonably independent-ish realization with no dominant
        periodicity to remove first - e.g. an isolated rotor in hover,
        where frame-to-frame turbulent decorrelation is the main effect
        and there's no strong once-per-revolution signal contaminating
        these statistics - or simply when dt/rpm aren't available.
    sync='revolution': only frames closest to a WHOLE number of
        revolutions (needs BOTH dt and rpm) - removes a periodic,
        once-per-revolution ripple that would otherwise stop a plain
        per-frame running mean/variance from cleanly asymptoting even
        once the run has genuinely converged (see cumulative_stats()'s
        original design note: stopping partway through a revolution
        includes an unequal, arbitrary slice of the periodic cycle each
        time - Pope's own asymptotic picture implicitly assumes each
        new sample is a comparable statistical realization, which for a
        periodic signal means one whole revolution, not one frame).
    sync='periodicity': only frames closest to a whole number of
        period_deg-sized steps (needs dt, rpm, AND period_deg) - a
        generalization of 'revolution' for a case whose physical
        periodicity is SHORTER than one full turn, e.g. a 4-blade rotor
        / 4-vane stator interaction that repeats every 360/4 = 90
        degrees in the stationary frame, not needing a full revolution
        to reach a comparable phase. period_deg=360 is identical to
        sync='revolution'.

    Parameters
    ----------
    n_frames : int
    dt : float, optional
        Physical timestep [s] between frames.
    rpm : float, optional
        Rotor speed [rev/min].
    sync : 'none', 'revolution', or 'periodicity'
    period_deg : float, optional
        Required only for sync='periodicity'.

    Returns
    -------
    idx : np.ndarray[int], shape (n_samples,)
        The frame indices to keep, in order - idx == arange(n_frames)
        for sync='none'.
    dt_eff : float or None
        Physical duration [s] BETWEEN two consecutive returned samples
        (dt itself for 'none', one full revolution period for
        'revolution', one period_deg-sized step for 'periodicity') -
        None if dt is None. Lets every caller convert a lag measured in
        "samples" into physical time consistently, whichever sync mode
        produced those samples.
    '''

    if sync == 'none':
        return np.arange(n_frames), dt

    if dt is None or rpm is None:
        raise ValueError(f"sync='{sync}' needs BOTH dt and rpm.")

    if sync == 'revolution':
        period_deg_eff = 360.0
    elif sync == 'periodicity':
        if period_deg is None:
            raise ValueError(
                "sync='periodicity' needs period_deg (e.g. 90.0 for a 4-blade rotor / "
                "4-vane stator interaction repeating every 360/4 degrees)."
            )
        period_deg_eff = period_deg
    else:
        raise ValueError(f"Unknown sync='{sync}' - use 'none', 'revolution', or 'periodicity'.")

    frames_per_period = (60.0 / (rpm * dt)) * (period_deg_eff / 360.0)
    n_periods = int(n_frames / frames_per_period)
    if n_periods < 1:
        raise ValueError(
            f"This series ({n_frames} frames) doesn't even span one full period "
            f"({frames_per_period:.1f} frames/period at rpm={rpm}, dt={dt}, "
            f"period_deg={period_deg_eff}) - can't sync."
        )

    idx = np.clip(np.round(np.arange(1, n_periods + 1) * frames_per_period).astype(int) - 1,
                  0, n_frames - 1)
    dt_eff = (period_deg_eff / 360.0) * (60.0 / rpm)

    return idx, dt_eff


def cumulative_mean(series, dt: float = None, rpm: float = None,
                     sync: str = 'none', period_deg: float = None):

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
    sync, period_deg : see _sync_indices() - 'none' (default, every
        frame), 'revolution' (needs dt+rpm), or 'periodicity' (needs
        dt+rpm+period_deg).

    Returns
    -------
    x : np.ndarray
        Frame index (if dt is None), else elapsed time [s] - one entry
        per SYNC'D sample, not necessarily one per raw frame.
    running_mean : np.ndarray, same shape as x
        Cumulative mean of the sync'd series up to and including each
        sample - running_mean[-1] equals the sync'd series' mean exactly
        (== series.mean() when sync='none').
    revolutions : np.ndarray, same shape as x, or None
        Revolutions elapsed at each sample - only computed if both dt
        and rpm are given.
    '''

    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"series must be 1D (one value per frame), got shape {series.shape}")
    if len(series) < 2:
        raise ValueError(f"Need at least 2 frames for a running mean - got {len(series)}.")

    idx, _ = _sync_indices(len(series), dt=dt, rpm=rpm, sync=sync, period_deg=period_deg)
    sub = series[idx]
    if len(sub) < 2:
        raise ValueError(f"Only {len(sub)} sample(s) left after sync='{sync}' - need at least 2.")

    running_mean = np.cumsum(sub) / np.arange(1, len(sub) + 1)
    x = idx * dt if dt is not None else idx.astype(float)

    revolutions = None
    if dt is not None and rpm is not None:
        revolutions = x * rpm / 60.0

    return x, running_mean, revolutions


def plot_cumulative_mean(series, dt: float = None, rpm: float = None, sync: str = 'none',
                          period_deg: float = None, ylabel: str = None,
                          ax=None, color: str = 'black', savepath: str = None, dpi: int = 600):

    '''
    Plot cumulative_mean()'s running mean vs. elapsed time (or frame
    index, if dt isn't given) - with a second x-axis in revolutions along
    the top if rpm is ALSO given, so "how many cycles until this
    flattens out" is readable directly rather than having to convert
    frame counts by hand.

    Parameters
    ----------
    series, dt, rpm, sync, period_deg : see cumulative_mean().
    ylabel : str, optional
        Defaults to a generic 'Cumulative mean' - pass the actual
        quantity's own label (e.g. 'Thrust [N]', matching
        StripForces._COMPONENT_LABELS' style) for a properly labeled
        axis.

    Returns
    -------
    (fig, ax)
    '''

    x, running_mean, revolutions = cumulative_mean(series, dt=dt, rpm=rpm, sync=sync, period_deg=period_deg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    marker = 'o-' if sync != 'none' else '-'
    ax.plot(x, running_mean, marker, color=color, linewidth=1.5, markersize=4)
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


def cumulative_stats(series, dt: float = None, rpm: float = None,
                      sync: str = 'none', period_deg: float = None):

    '''
    Cumulative (running) MEAN and VARIANCE of a scalar time series as a
    function of how many samples have been included so far - mirrors
    Pope, "Turbulent Flows" (2000)'s own illustration of <U> and <u'^2>
    both approaching an asymptote as more samples/time are included, for
    a statistically stationary process (no specific page cited - verify
    against the book's own index).

    sync, period_deg : see _sync_indices() - 'none' (default, every
        frame - appropriate when each frame is already a reasonably
        independent-ish realization, e.g. an isolated rotor in hover),
        'revolution' (needs dt+rpm - removes a once-per-revolution
        ripple), or 'periodicity' (needs dt+rpm+period_deg - removes a
        shorter, sub-revolution ripple, e.g. a 4-blade/4-vane rotor-
        stator's 90-degree interaction period). Without syncing to
        whatever the case's real periodicity actually is, a plain
        per-frame running mean/variance of a periodic-plus-turbulent
        signal does NOT cleanly asymptote even once the run has
        genuinely converged - it shows a persistent, phase-dependent
        RIPPLE, since stopping partway through a cycle includes an
        unequal, arbitrary slice of it each time. If you DON'T see an
        asymptote even with the correct sync mode on, that's now much
        more likely a genuine "not converged yet" finding than a
        plotting artifact.

    Parameters
    ----------
    series, dt, rpm : see cumulative_mean().

    Returns
    -------
    x : np.ndarray
        Frame index or elapsed time [s] (see cumulative_mean()) - one
        entry per sync'd sample.
    running_mean, running_var : np.ndarray, same shape as x
        running_var[-1] equals the sync'd series' var() (population/
        biased variance - E[X^2] - E[X]^2 - the fluctuation-about-the-
        mean convention already used elsewhere in this project, e.g.
        FrictionLines.cf(stat='rms')) exactly, same guarantee as
        cumulative_mean()'s running_mean[-1].
    revolutions : np.ndarray, or None - see cumulative_mean().
    '''

    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"series must be 1D (one value per frame), got shape {series.shape}")
    if len(series) < 2:
        raise ValueError(f"Need at least 2 frames for running stats - got {len(series)}.")

    idx, _ = _sync_indices(len(series), dt=dt, rpm=rpm, sync=sync, period_deg=period_deg)
    sub = series[idx]
    if len(sub) < 2:
        raise ValueError(f"Only {len(sub)} sample(s) left after sync='{sync}' - need at least 2.")

    n = np.arange(1, len(sub) + 1)
    running_mean = np.cumsum(sub) / n
    running_meansq = np.cumsum(sub ** 2) / n
    running_var = running_meansq - running_mean ** 2

    x = idx * dt if dt is not None else idx.astype(float)
    revolutions = x * rpm / 60.0 if (dt is not None and rpm is not None) else None

    return x, running_mean, running_var, revolutions


def plot_cumulative_stats(series, dt: float = None, rpm: float = None, sync: str = 'none',
                           period_deg: float = None, ylabel: str = None, color: str = 'black',
                           savepath: str = None, dpi: int = 600):

    '''
    Two-panel plot of cumulative_stats()'s running mean (top) and running
    variance (bottom) vs. elapsed time/revolutions - directly mirrors
    Pope's own <U>-and-<u'^2>-vs-sample-count figure (see that function's
    docstring for the citation caveat and what sync/period_deg fix for a
    periodic-plus-turbulent signal).

    Returns
    -------
    (fig, (ax_mean, ax_var))
    '''

    x, running_mean, running_var, revolutions = cumulative_stats(
        series, dt=dt, rpm=rpm, sync=sync, period_deg=period_deg)

    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    marker = 'o-' if sync != 'none' else '-'
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


def cumulative_moments(series, dt: float = None, rpm: float = None,
                        sync: str = 'none', period_deg: float = None):

    '''
    Cumulative (running) SKEWNESS and FLATNESS (kurtosis) of a scalar
    time series - the THIRD and FOURTH standardized moments, extending
    cumulative_stats()'s mean/variance (1st/2nd order) to check whether
    these higher-order statistics have ALSO stopped changing:

        skewness = mu3 / mu2^1.5      (0 for a symmetric distribution)
        flatness = mu4 / mu2^2        (3 for a Gaussian - "excess
                                        kurtosis" is flatness - 3)

    where mu2/mu3/mu4 are the 2nd/3rd/4th CENTRAL moments (about the
    running mean). Higher moments are disproportionately sensitive to
    rare, large-amplitude events in a distribution's tail (a single big
    excursion barely moves the mean, moves the variance a bit, and can
    swing skewness/flatness a lot), so they are well known to need
    substantially more samples to converge than the mean or variance do
    - Pope, "Turbulent Flows", and Tennekes & Lumley, "A First Course in
    Turbulence", both discuss skewness/flatness as standard turbulence
    statistics (no specific page cited for either - verify against each
    book's own index before citing a page number).

    Numerically: raw statistical moments of a large-mean, small-
    fluctuation signal (e.g. thrust ~1.3 N with sub-percent turbulent
    fluctuations) are prone to catastrophic cancellation if accumulated
    directly from `series` (the central moment is then a SMALL
    difference between LARGE raw-moment numbers). Avoided here by
    shifting the sync'd series once, up front, by its own overall mean
    (skewness/flatness are exactly invariant to a location shift) before
    accumulating any running power sums - not a numerically-optimal
    single-pass algorithm (a true one-pass stable higher-moment update,
    e.g. Pebay (2008), would avoid even this), but validated here
    against scipy.stats.skew/kurtosis on synthetic data and sufficient
    for this project's actual signal magnitudes.

    sync, period_deg : see cumulative_stats() / _sync_indices() - same
        three modes ('none', 'revolution', 'periodicity').

    Returns
    -------
    x, revolutions : see cumulative_stats().
    running_skewness, running_flatness : np.ndarray, same shape as x -
        the final entry of each equals the standard (biased, population)
        skewness/kurtosis of the full sync'd series, matching
        scipy.stats.skew(..., bias=True) / scipy.stats.kurtosis(...,
        fisher=False, bias=True) exactly (validated).
    '''

    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"series must be 1D (one value per frame), got shape {series.shape}")

    idx, _ = _sync_indices(len(series), dt=dt, rpm=rpm, sync=sync, period_deg=period_deg)
    sub = series[idx]
    if len(sub) < 4:
        raise ValueError(
            f"Only {len(sub)} sample(s) left after sync='{sync}' - need at least 4 for a "
            "meaningful running skewness/flatness."
        )

    shift = sub.mean()
    s = sub - shift  # location shift only - doesn't change skewness/flatness, avoids cancellation

    n = np.arange(1, len(s) + 1)
    m1 = np.cumsum(s) / n
    m2 = np.cumsum(s ** 2) / n
    m3 = np.cumsum(s ** 3) / n
    m4 = np.cumsum(s ** 4) / n

    mu2 = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4

    with np.errstate(invalid='ignore', divide='ignore'):
        running_skewness = mu3 / mu2 ** 1.5
        running_flatness = mu4 / mu2 ** 2

    x = idx * dt if dt is not None else idx.astype(float)
    revolutions = x * rpm / 60.0 if (dt is not None and rpm is not None) else None

    return x, running_skewness, running_flatness, revolutions


def plot_cumulative_moments(series, dt: float = None, rpm: float = None, sync: str = 'none',
                             period_deg: float = None, label: str = None, color: str = 'black',
                             savepath: str = None, dpi: int = 600):

    '''
    Two-panel plot of cumulative_moments()'s running skewness (top) and
    running flatness (bottom) vs. elapsed time/revolutions - the same
    idea as plot_cumulative_stats(), one order higher.

    label : str, optional
        Prefixed to the y-axis labels (e.g. 'Thrust') - defaults to
        nothing (plain 'Skewness [-]' / 'Flatness [-]').

    Returns
    -------
    (fig, (ax_skew, ax_flat))
    '''

    x, running_skewness, running_flatness, revolutions = cumulative_moments(
        series, dt=dt, rpm=rpm, sync=sync, period_deg=period_deg)

    fig, (ax_skew, ax_flat) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    marker = 'o-' if sync != 'none' else '-'
    prefix = f'{label} ' if label else ''

    ax_skew.plot(x, running_skewness, marker, color=color, linewidth=1.5, markersize=4)
    ax_skew.axhline(0, color='k', linewidth=0.8, alpha=0.5, linestyle='--')
    ax_skew.set_ylabel(f'{prefix}Skewness [-]')
    ax_skew.grid(True, alpha=0.4)

    ax_flat.plot(x, running_flatness, marker, color=color, linewidth=1.5, markersize=4)
    ax_flat.axhline(3, color='k', linewidth=0.8, alpha=0.5, linestyle='--')
    ax_flat.set_xlabel('Time [s]' if dt is not None else 'Frame index')
    ax_flat.set_ylabel(f'{prefix}Flatness [-]')
    ax_flat.grid(True, alpha=0.4)
    ax_flat.set_xlim(x[0], x[-1])

    if revolutions is not None:
        ax_top = ax_skew.twiny()
        ax_top.set_xlim(revolutions[0], revolutions[-1])
        ax_top.set_xlabel('Revolutions included')

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, (ax_skew, ax_flat)


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
    plot_autocorrelation_windows() - or integrating it into a single
    correlation timescale - see integral_timescale()/standard_error().

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


def integral_timescale(series, dt: float = None, rpm: float = None, sync: str = 'none',
                        period_deg: float = None, max_lag: int = None):

    '''
    Integral timescale T_int = integral of rho(s) ds from s=0 up to (and
    including) its first zero-crossing - the standard practical
    truncation for this integral (Flyvbjerg, H., & Petersen, H. G.,
    "Error estimates on averages of correlated data", Journal of
    Chemical Physics, 91(1), 461-466, 1989, use exactly this kind of
    correlation-time argument to put honest error bars on the mean of a
    correlated time series - originally from molecular dynamics, but the
    statistics are identical for any autocorrelated stochastic
    simulation trace). Integrating rho(s) past its first zero crossing
    would just accumulate its noise floor (see autocorrelation()'s
    docstring on the ~1/sqrt(N) scatter expected there for an
    uncorrelated remainder), biasing T_int upward - stopping at the
    first zero crossing is the standard fix.

    sync/period_deg (see _sync_indices()): for a signal with a real
    periodic component (once-per-revolution, or once-per-period_deg),
    the RAW per-frame autocorrelation never cleanly decays to zero - it
    keeps oscillating at the periodic frequency, which would badly
    corrupt a "first zero crossing" search. Passing sync='revolution'
    (or 'periodicity') first collapses the series to one value per
    cycle, so the correlation being measured here is genuine cycle-to-
    cycle decorrelation, not the periodicity itself. Use sync='none'
    only when confident there's no dominant periodic component (e.g.
    this project's isolated-rotor-in-hover case, where each frame is
    already a roughly independent-ish realization) - see README.md's
    convergence-checking section.

    Returns
    -------
    T_int : float
        Integral timescale - in SECONDS if dt is given (converted using
        the correct per-sample duration for whichever sync mode was
        used), else in raw sample units.
    lags, rho : see autocorrelation() - the underlying estimate (on the
        SYNC'D series), for inspection/plotting.
    '''

    series = np.asarray(series, dtype=float)
    idx, dt_eff = _sync_indices(len(series), dt=dt, rpm=rpm, sync=sync, period_deg=period_deg)
    sub = series[idx]

    lags, rho = autocorrelation(sub, max_lag=max_lag)

    sign_changes = np.where(np.diff(np.sign(rho)) < 0)[0]  # first + -> - crossing
    cutoff = sign_changes[0] + 1 if len(sign_changes) > 0 else len(rho) - 1

    T_int_samples = np.trapz(rho[:cutoff + 1], lags[:cutoff + 1])
    T_int = T_int_samples * dt_eff if dt_eff is not None else T_int_samples

    return T_int, lags, rho


def standard_error(series, dt: float = None, rpm: float = None, sync: str = 'none',
                    period_deg: float = None, max_lag: int = None):

    '''
    Estimated standard error of the mean (SEM) of `series`'s time
    average, from the classical result for a stationary random process
    (T = total averaging time, T_int = integral_timescale()):

        Var[time-average] ~= 2 * sigma^2 * T_int / T      (valid for T >> T_int)
        SEM = sigma * sqrt(2 * T_int / T)

    equivalently, in terms of an EFFECTIVE number of independent samples
    N_eff = T / (2*T_int):

        SEM = sigma / sqrt(N_eff)

    - the ordinary "standard error = sigma/sqrt(n)" formula, but with n
    replaced by however many independent-equivalent realizations your
    correlated time series actually contains, rather than its raw frame
    count. See integral_timescale()'s docstring for the literature this
    is based on and its own sync/periodicity caveats (passed straight
    through here).

    IMPORTANT: only valid for T >> T_int (see integral_timescale()), and
    only once the signal is past its initial transient - sigma/T_int
    estimated across a transient are both meaningless for THIS formula,
    which assumes stationarity. See README.md's convergence-checking
    section on discarding a warm-up window before calling this.

    Returns
    -------
    dict with keys:
        mean : float - mean of the sync'd series.
        sigma : float - std dev of the sync'd series (population).
        T_int : float - see integral_timescale() (seconds if dt given).
        T : float - total physical duration actually simulated [s] if
            dt is given, else the raw frame count - ALWAYS the FULL,
            un-sync'd series' duration (T is how much real simulation
            time you've actually run, regardless of how finely sync
            subsamples it for the correlation estimate).
        n_eff : float - T / (2*T_int) - effective independent samples.
        sem : float - standard error of the mean, same units as series.
        relative_sem : float or None - sem / |mean|, None if mean == 0.
    '''

    series = np.asarray(series, dtype=float)
    idx, _ = _sync_indices(len(series), dt=dt, rpm=rpm, sync=sync, period_deg=period_deg)
    sub = series[idx]

    mean = sub.mean()
    sigma = sub.std()
    T_int, _, _ = integral_timescale(series, dt=dt, rpm=rpm, sync=sync,
                                      period_deg=period_deg, max_lag=max_lag)

    T = len(series) * dt if dt is not None else float(len(series))

    n_eff = T / (2.0 * T_int) if T_int > 0 else np.inf
    sem = sigma / np.sqrt(n_eff) if np.isfinite(n_eff) and n_eff > 0 else np.nan
    relative_sem = sem / abs(mean) if mean != 0 else None

    return {
        'mean': mean, 'sigma': sigma, 'T_int': T_int, 'T': T,
        'n_eff': n_eff, 'sem': sem, 'relative_sem': relative_sem,
    }


def plot_integral_timescale(series, dt: float = None, rpm: float = None, sync: str = 'none',
                             period_deg: float = None, max_lag: int = None,
                             target_sem: float = None, target_relative_sem: float = 0.01,
                             ax=None, color: str = 'black', savepath: str = None, dpi: int = 600):

    '''
    Plot the (sync'd) autocorrelation rho(s) underlying integral_timescale()/
    standard_error(), with the region actually integrated (s=0 up to the
    first zero-crossing) shaded, and an upper-left inset box (same style
    as StripForces.plot_bar_forces()'s show_totals) reporting T_int, SEM,
    n_eff, AND - if dt and rpm are both available - required_averaging_time()'s
    revolutions_required/additional_revolutions for a target SEM: the
    actual "how many more revolutions do I need" answer, not just the
    correlation curve it's derived from.

    target_sem, target_relative_sem : see required_averaging_time() -
        give at most one; target_relative_sem defaults to 0.01 (1% of
        the mean) so the revolutions line shows up out of the box.
        Neither is used (and the inset falls back to just T_int/SEM/
        n_eff) if dt is None, since required_averaging_time() needs a
        real timestep.

    Returns
    -------
    (fig, ax)
    '''

    T_int, lags, rho = integral_timescale(series, dt=dt, rpm=rpm, sync=sync,
                                           period_deg=period_deg, max_lag=max_lag)
    stats = standard_error(series, dt=dt, rpm=rpm, sync=sync, period_deg=period_deg, max_lag=max_lag)

    _, dt_eff = _sync_indices(len(np.asarray(series)), dt=dt, rpm=rpm, sync=sync, period_deg=period_deg)
    x_lag = lags * dt_eff if dt_eff is not None else lags

    sign_changes = np.where(np.diff(np.sign(rho)) < 0)[0]
    cutoff = sign_changes[0] + 1 if len(sign_changes) > 0 else len(rho) - 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    ax.plot(x_lag, rho, '-', color=color, linewidth=1.5)
    ax.fill_between(x_lag[:cutoff + 1], rho[:cutoff + 1], 0, color=color, alpha=0.15,
                     label='Integrated region')
    ax.axhline(0, color=color, linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Lag [s]' if dt_eff is not None else 'Lag [samples]')
    ax.set_ylabel(r'$\rho(s)$ [-]')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')

    unit = 's' if dt_eff is not None else 'samples'
    note = (
        f"$T_{{int}}$ = {T_int:.4g} {unit}\n"
        f"SEM = {stats['sem']:.4g}\n"
        f"$n_{{eff}}$ = {stats['n_eff']:.4g}"
    )

    if dt is not None:
        req = required_averaging_time(
            series, dt=dt, rpm=rpm, sync=sync, period_deg=period_deg, max_lag=max_lag,
            target_sem=target_sem, target_relative_sem=None if target_sem is not None else target_relative_sem,
        )
        target_pct = target_sem if target_sem is not None else f"{target_relative_sem * 100:.3g}\\%"
        if req['revolutions_required'] is not None:
            note += (
                f"\nRevs needed ({target_pct} target) = {req['revolutions_required']:.4g}\n"
                f"(+{req['additional_revolutions']:.4g} more)"
            )
        else:
            note += f"\n$T_{{required}}$ ({target_pct} target) = {req['T_required']:.4g} s"

    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=18, va='top', ha='left',
            zorder=10, bbox=dict(facecolor='white', edgecolor='black', alpha=1.0))

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax


def required_averaging_time(series, dt: float, rpm: float = None, sync: str = 'none',
                             period_deg: float = None, target_sem: float = None,
                             target_relative_sem: float = None, max_lag: int = None):

    '''
    Inverts standard_error()'s formula to answer "how much MORE
    averaging time (or how many more revolutions, if rpm is given) is
    needed for the mean's standard error to reach a target precision":

        T_required = 2 * sigma^2 * T_int / target_sem^2

    using this series' OWN sigma/T_int as the estimate - so this is only
    as reliable as those estimates are (see standard_error()'s caveats:
    needs T >> T_int already, and a signal past its initial transient -
    a short/still-transient run will give an unreliable, possibly wildly
    wrong, T_required).

    Parameters
    ----------
    series : array-like, shape (n_frames,)
    dt : float
        Required (unlike elsewhere in this module) - a physical
        averaging TIME is meaningless without one.
    target_sem : float, optional
        Absolute target standard error, same units as `series`. Give
        EXACTLY ONE of target_sem/target_relative_sem.
    target_relative_sem : float, optional
        Target standard error as a fraction of |mean| (e.g. 0.001 for
        0.1%).

    Returns
    -------
    dict with keys:
        T_required : float [s] - total averaging time needed.
        T_current : float [s] - what you already have.
        additional_T : float [s] - max(0, T_required - T_current), the
            actual "how much MORE" answer; 0 if already sufficient.
        revolutions_required, revolutions_current, additional_revolutions :
            same, in revolutions - None if rpm not given.
        stats : standard_error()'s own return dict, for inspection.
    '''

    if dt is None:
        raise ValueError("required_averaging_time() needs dt - a physical time target needs a real timestep.")
    if (target_sem is None) == (target_relative_sem is None):
        raise ValueError("Give exactly one of target_sem or target_relative_sem.")

    stats = standard_error(series, dt=dt, rpm=rpm, sync=sync, period_deg=period_deg, max_lag=max_lag)

    if target_relative_sem is not None:
        if stats['mean'] == 0:
            raise ValueError("target_relative_sem needs a nonzero series mean.")
        sem_target = target_relative_sem * abs(stats['mean'])
    else:
        sem_target = target_sem

    T_required = 2.0 * stats['sigma'] ** 2 * stats['T_int'] / sem_target ** 2
    T_current = stats['T']
    additional_T = max(0.0, T_required - T_current)

    revolutions_required = T_required * rpm / 60.0 if rpm is not None else None
    revolutions_current = T_current * rpm / 60.0 if rpm is not None else None
    additional_revolutions = additional_T * rpm / 60.0 if rpm is not None else None

    return {
        'T_required': T_required, 'T_current': T_current, 'additional_T': additional_T,
        'revolutions_required': revolutions_required, 'revolutions_current': revolutions_current,
        'additional_revolutions': additional_revolutions, 'stats': stats,
    }


def cycle_correlation(series, dt: float, rpm: float, period_deg: float = 360.0, n_points: int = 100):

    '''
    Cycle-to-cycle Pearson correlation of a scalar time series, folded
    onto consecutive cycles of period_deg degrees (360, the default =
    one full revolution; e.g. 90 for a 4-blade rotor / 4-vane stator
    interaction repeating every 360/4 degrees - same period_deg
    convention as _sync_indices()'s sync='periodicity').

    This checks a DIFFERENT thing than cumulative_mean()/
    cumulative_stats()/cumulative_moments(): not whether a running
    STATISTIC has flattened, but whether the WAVEFORM SHAPE ITSELF has
    stopped changing from one cycle to the next - approaching a
    correlation of 1 as consecutive cycles become indistinguishable.
    This is the assumption StripForces.phase_lock()/harmonics() and
    TipVortexPhaseAverage's phase-locked averaging already depend on
    (that the per-revolution waveform is repeatable), so this directly
    validates it rather than checking a generic turbulence statistic -
    standard practice in time-accurate rotor CFD (checking cycle-to-
    cycle periodicity/repeatability before trusting a phase-locked
    average or extracting acoustic/performance data from it), though no
    single specific paper+page is cited here.

    Each cycle is resampled (linear interpolation) onto a common
    `n_points`-point phase grid BEFORE correlating, since a cycle's
    frame count is rarely a whole number (frames_per_cycle =
    60/(rpm*dt) * period_deg/360) - two cycles are compared at the SAME
    phase values, not just the same number of raw frames.

    Parameters
    ----------
    series : array-like, shape (n_frames,)
    dt, rpm : float
        Required - there is no "sync='none'" equivalent here; there's
        no notion of a "cycle" to correlate without a real period.
    period_deg : float
        Cycle length in degrees - 360 (default) for one full revolution,
        or a case's own shorter geometric periodicity.
    n_points : int
        Phase-grid resolution used for the resampling/correlation.

    Returns
    -------
    cycle_index : np.ndarray, shape (n_cycles - 1,)
        1-indexed FROM THE SECOND cycle - cycle_index[0] == 2 means
        "cycle 1 vs. cycle 2".
    correlation : np.ndarray, same shape - Pearson r between consecutive
        cycles, in [-1, 1] - -> 1 as the waveform stops changing.
    '''

    series = np.asarray(series, dtype=float)
    if dt is None or rpm is None:
        raise ValueError("cycle_correlation() needs both dt and rpm.")

    frames_per_cycle = (60.0 / (rpm * dt)) * (period_deg / 360.0)
    n_cycles = int(len(series) / frames_per_cycle)
    if n_cycles < 2:
        raise ValueError(
            f"This series ({len(series)} frames) spans fewer than 2 full cycles "
            f"({frames_per_cycle:.1f} frames/cycle at rpm={rpm}, dt={dt}, period_deg={period_deg}) "
            "- need at least 2 to correlate."
        )

    phase_grid = np.linspace(0, frames_per_cycle, n_points, endpoint=False)
    frame_axis = np.arange(len(series))

    waveforms = []
    for c in range(n_cycles):
        frame_positions = c * frames_per_cycle + phase_grid
        if frame_positions[-1] > len(series) - 1:
            break
        waveforms.append(np.interp(frame_positions, frame_axis, series))
    waveforms = np.array(waveforms)

    if len(waveforms) < 2:
        raise ValueError("Fewer than 2 complete cycles fit in this series - can't correlate.")

    cycle_index = np.arange(2, len(waveforms) + 1)
    correlation = np.array([
        np.corrcoef(waveforms[i], waveforms[i + 1])[0, 1] for i in range(len(waveforms) - 1)
    ])

    return cycle_index, correlation


def plot_cycle_correlation(series, dt: float, rpm: float, period_deg: float = 360.0, n_points: int = 100,
                            ax=None, color: str = 'black', savepath: str = None, dpi: int = 600):

    '''
    Plot cycle_correlation()'s cycle-to-cycle correlation vs. cycle
    index - a dashed line at 1 marks perfect cycle-to-cycle repeatability
    for reference.

    Returns
    -------
    (fig, ax)
    '''

    cycle_index, correlation = cycle_correlation(series, dt, rpm, period_deg=period_deg, n_points=n_points)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    ax.plot(cycle_index, correlation, 'o-', color=color, linewidth=1.5, markersize=5)
    ax.axhline(1.0, color='k', linewidth=0.8, alpha=0.5, linestyle='--')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Cycle-to-cycle correlation [-]')
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax
