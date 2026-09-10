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
                          ax=None, color: str = 'tab:blue', savepath: str = None, dpi: int = 150):

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
