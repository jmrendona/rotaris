'''
Meridional-plane (fixed-azimuth radial/axial cut) field plotter for a
PowerVIZ "Matrix" export - a single regular (ny, nx) grid of one scalar
field per file (NOT the scattered x,y,z,value format used elsewhere in
this project - see such a file's own 15-line header:
"# Format: Matrix of <Variable>").

Axis convention (confirmed for this project's isolated-rotor-in-hover
cases, e.g. 6e-5_6000rpm - see HANDOFF.md's meridional-plane
investigation): at the 0deg-azimuth plane, raw X-Velocity is RADIAL,
Y-Velocity is AXIAL, Z-Velocity is TANGENTIAL (matches cases/*.yaml's
own axes.thickness: 1 - the rotor's rotation/axial axis is Y).

For 6e-5_6000rpm's own txt/*_Ndeg.txt files, "Ndeg" DOES mean the plane
was physically swept to that azimuth, and raw X/Y/Z-Velocity need
meridional_cylindrical_components() before comparing/overlaying across
angle files - confirmed empirically (X-Velocity taken naively as
"radial" at every angle, with no rotation at all, visibly looks
tangential rather than radial by 60/90/120deg).

Two dead ends hit while pinning this down, worth knowing about before
trusting either kind of "proof" again on a future dataset:
- Each file's own header (Slice Normal / Red Direction / Green
  Direction / Center Point) looks numerically IDENTICAL (~1e-17) across
  every "Ndeg" file, which looks like strong evidence the plane never
  moves - it isn't. That header is apparently reported in each slice's
  own local in-plane basis, so it trivially looks the same regardless
  of the slice's true orientation; it says nothing about whether the
  plane is swept. Don't use it to decide either way.
- txt/azimuth_vel_Ndeg.txt looks like it should be PowerVIZ's own
  independently-computed ground-truth tangential velocity, useful for
  checking the rotation sign against. It isn't, for this dataset at
  least: azimuth_vel_30deg.txt is bit-identical to z_vel_30deg.txt
  (confirmed via direct array comparison, not just a text diff - a
  first check misread a piped `diff | head`'s exit code as diff's own
  and got this backwards). Don't treat it as an independent check.

Rotation SIGN needs meridional_cylindrical_components(..., angle_deg=
-angle_deg) for this dataset (negated, not the raw "Ndeg" value) -
confirmed empirically by which sign makes Ur/Ut visually keep their own
identity (dipole-at-vortex-core vs. broad antisymmetric swirl) across
angles instead of swapping - see tip_vortex_manager.py's own usage.
This sign is this dataset's own finding (PowerVIZ's sweep direction vs.
cylindrical_velocity's rotation sense), not a general property of
meridional_cylindrical_components() - re-check against a reference
figure for any other dataset rather than assuming -angle_deg carries
over.
'''

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap

from bladeprocessor.tip_vortex_tracking import cylindrical_velocity

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18
})

# Hand-built diverging colormap carried over from the original
# (non-rotaris) version of this plotter - kept as an opt-in (pass
# cmap='custom' to plot_meridional_field) - DEFAULT is 'RdBu_r',
# matching this project's own convention for a signed/diverging
# physical field elsewhere (bladeprocessor/pod.py's plot_pod_mode(),
# the LSB vorticity plots).
_CUSTOM_NODES = [0, 0.111111, 0.11465, 0.36508, 0.492063, 0.601911, 0.619048, 0.873016, 1]
_CUSTOM_COLORS = [
    (0, 0, 0.5625), (0, 0, 1), (0, 0.0117647, 1), (0, 1, 1), (0.5, 1, 0.5),
    (0.941176, 1, 0.0588235), (1, 1, 0), (1, 0, 0), (0.5, 0, 0),
]
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom_map', list(zip(_CUSTOM_NODES, _CUSTOM_COLORS)))


def load_meridional_matrix(data_file: str, lx: float = None, ly: float = None):

    '''
    Parse one PowerVIZ Matrix-format .txt export into physical (X, Y,
    data) arrays, in METERS - lx/ly, if not given, are read directly
    from the file's own 15-line header ("# Size: (lx, ly)"), so there's
    no need to pass them by hand unless overriding.

    Returns
    -------
    X, Y : np.ndarray, shape (ny, nx), meshgrid, meters
    data : np.ndarray, shape (ny, nx)
    '''

    with open(data_file) as f:
        header_lines = [next(f) for _ in range(15)]

    if lx is None or ly is None:
        size_line = [l for l in header_lines if l.startswith('# Size:')][0]
        lx_file, ly_file = (float(v) for v in size_line.split('(')[1].split(')')[0].split(','))
        lx = lx_file if lx is None else lx
        ly = ly_file if ly is None else ly

    data = np.genfromtxt(data_file, skip_header=15, filling_values=np.nan)
    ny, nx = data.shape
    x = np.linspace(-lx / 2, lx / 2, nx)
    y = np.linspace(-ly / 2, ly / 2, ny)
    X, Y = np.meshgrid(x, y)

    return X, Y, data


def meridional_cylindrical_components(x_vel_file: str, y_vel_file: str, z_vel_file: str,
                                       angle_deg: float, rotation_axis: str = 'y',
                                       lx: float = None, ly: float = None):

    '''
    Load one azimuth's raw x/y/z-velocity Matrix exports (same one-sided
    meridional plane, all three sharing the same grid - see this
    module's docstring) and project them onto cylindrical (v_r, v_theta,
    v_z) via tip_vortex_tracking.cylindrical_velocity(), so this plane
    can be directly compared/overlaid against the 0deg plane (or any
    other azimuth) on the same physical radial/axial basis.

    Before trusting this on a NEW set of "_Ndeg.txt" files, verify the
    rotation empirically against a reference figure - don't rely on the
    files' own header (Slice Normal / Red Direction / Green Direction /
    Center Point) to decide whether the planes are swept; that header is
    apparently reported in each slice's own local basis and looks
    numerically identical across every angle regardless of the plane's
    true orientation (confirmed misleading for 6e-5_6000rpm's own
    txt/*_Ndeg.txt files - see this module's own docstring). Also check
    the rotation SIGN empirically (plot a couple of angles and confirm
    v_r/v_theta each keep their own character rather than swapping) -
    6e-5_6000rpm needed angle_deg=-angle_deg, not the raw "Ndeg" value;
    that sign is dataset-specific, not something this function can get
    right by construction.

    Only the pure coordinate rotation runs here - omega_rad_s is NOT
    exposed (left None in the underlying call), same as
    tip_vortex_manager.py's own usage of cylindrical_velocity(): whether
    this pipeline's velocity is absolute or already LRF-relative has not
    been verified (see tip_vortex_tracking.py's docstring), so no frame
    correction is applied here either.

    Parameters
    ----------
    x_vel_file, y_vel_file, z_vel_file : str
        Paths to this azimuth's raw X/Y/Z-Velocity Matrix exports (e.g.
        x_vel_30deg.txt, y_vel_30deg.txt, z_vel_30deg.txt).
    angle_deg : float
        This plane's azimuth, in the same convention as
        cylindrical_velocity's own angle_deg (see that function's
        docstring for the exact rotation sign convention).
    rotation_axis : str
        Rotor rotation axis ('x', 'y', or 'z'), i.e. which raw component
        is the AXIAL one and therefore untouched by the rotation -
        default 'y', matching this project's own isolated-rotor-in-hover
        convention (cases/*.yaml's axes.thickness: 1). Pass the
        equivalent of cfg.thickness_axis (0/1/2 -> 'x'/'y'/'z') if a
        case uses a different mapping.

    Returns
    -------
    X, Y : np.ndarray, shape (ny, nx), meshgrid, meters (this plane's
        OWN in-plane grid - not yet divided by r_tip).
    v_r, v_theta, v_z : np.ndarray, shape (ny, nx)
    '''

    X, Y, vx = load_meridional_matrix(x_vel_file, lx, ly)
    _, _, vy = load_meridional_matrix(y_vel_file, lx, ly)
    _, _, vz = load_meridional_matrix(z_vel_file, lx, ly)

    if vy.shape != vx.shape or vz.shape != vx.shape:
        raise ValueError(
            f"x/y/z-velocity files don't share the same grid shape: "
            f"{vx.shape}, {vy.shape}, {vz.shape}."
        )

    v_r, v_theta, v_z = cylindrical_velocity(vx, vy, vz, angle_deg, axis=rotation_axis)

    return X, Y, v_r, v_theta, v_z


def plot_meridional_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, variable_label: str,
                           r_tip: float, normalize_by: float = None, sign_flip: bool = False,
                           vmin: float = None, vmax: float = None, n_levels: int = 101,
                           cmap: str = 'RdBu_r', xlim: tuple = None, ylim: tuple = None,
                           equal_aspect: bool = False, ax=None, savepath: str = None, dpi: int = 600):

    '''
    Filled-contour plot of an already-loaded meridional field (from
    load_meridional_matrix() or meridional_cylindrical_components()), in
    r/R, y/R coordinates - same style/role as
    tip_vortex_tracking.TipVortexPhaseAverage.plot_age_label(), but for
    this pipeline's regular-grid Matrix exports instead of a scattered
    point cloud.

    Parameters
    ----------
    X, Y : np.ndarray, shape (ny, nx)
        In-plane coordinates, in METERS (as returned by
        load_meridional_matrix() / meridional_cylindrical_components() -
        divided by r_tip internally, not before calling).
    Z : np.ndarray, shape (ny, nx)
        The field to plot.
    variable_label : str
        Colorbar label, e.g. r'mean $U_a$ [m/s]' - this function does
        NOT know or assume which physical quantity Z actually is.
    normalize_by : float, optional
        Divide Z by this (e.g. blade tip speed) before plotting - None
        (default) plots Z unchanged.
    sign_flip : bool
        Negate Z before plotting - off by default. Some PowerVIZ sign
        conventions come out inverted relative to the physical
        convention wanted for a given figure; check against a reference
        figure before relying on this rather than assuming either way.
    cmap : str
        'RdBu_r' (default) or 'custom' for CUSTOM_CMAP (this module's
        own hand-built diverging colormap, also importable directly).
    ax : matplotlib Axes, optional
        Plot into this axes instead of creating a new figure (e.g. for a
        multi-panel comparison across azimuths).

    Returns
    -------
    (fig, ax)
    '''

    Zp = Z.copy()
    if sign_flip:
        Zp = -Zp
    if normalize_by is not None:
        Zp = Zp / normalize_by

    if vmin is None:
        vmin = np.nanmin(Zp)
    if vmax is None:
        vmax = np.nanmax(Zp)
    levels = np.linspace(vmin, vmax, n_levels)
    used_cmap = CUSTOM_CMAP if cmap == 'custom' else cmap

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    cf = ax.contourf(X / r_tip, Y / r_tip, Zp, levels=levels, cmap=used_cmap,
                      vmin=vmin, vmax=vmax, extend='both')
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(variable_label)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('$r/R$')
    ax.set_ylabel('$y/R$')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # NOT equal-aspect by default - the original reference image this
    # was rebuilt from isn't physically square either (plain contourf,
    # no aspect call) - pass equal_aspect=True for true physical
    # proportions instead.
    if equal_aspect:
        ax.set_aspect('equal')
    fig.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=dpi)

    return fig, ax


def plot_meridional_component(data_file: str, variable_label: str, r_tip: float,
                               lx: float = None, ly: float = None, **plot_kwargs):

    '''
    Convenience wrapper: load ONE raw Matrix file directly and plot it -
    for a field that needs no cylindrical projection at all (e.g.
    pressure, vorticity magnitude), or for the 0deg plane specifically,
    where raw X/Y/Z-Velocity already ARE radial/axial/tangential (see
    this module's docstring - NOT true at any other azimuth; use
    meridional_cylindrical_components() + plot_meridional_field()
    instead for those).

    **plot_kwargs forwarded to plot_meridional_field() (normalize_by,
    sign_flip, vmin/vmax, cmap, xlim/ylim, equal_aspect, ax, savepath,
    dpi).

    Returns
    -------
    (fig, ax, X, Y, Z) - X, Y in meters (not yet divided by r_tip), Z
    the raw field (before sign_flip/normalize_by).
    '''

    X, Y, Z = load_meridional_matrix(data_file, lx, ly)
    fig, ax = plot_meridional_field(X, Y, Z, variable_label, r_tip, **plot_kwargs)
    return fig, ax, X, Y, Z
