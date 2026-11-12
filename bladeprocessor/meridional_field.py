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


# ------------- Scattered (x, y, z, value) meridional exports - see module docstring's own "Second
# dataset" note below for the file format and axis-convention differences from the Matrix-format
# functions above ------------- #
#
# A SECOND, GEOMETRICALLY DIFFERENT dataset (this project's own tip-vortex-tracking "_PL" case,
# e.g. 6e-5_6000rpm_PL/data/<component>/<name>_<N>deg_avg.txt) uses PowerVIZ's SCATTERED
# "# Format: position x y z value(<Variable>)" export (NOT the regular-grid Matrix format above),
# AND - confirmed empirically, do NOT assume this carries over from the Matrix-format dataset
# above or vice versa - this plane genuinely IS swept to each file's own azimuth in 3D space:
# at 0deg, X spans the full diameter and Z is a near-zero (+-0.001 m) slice-thickness residual;
# at 90deg, it's reversed (Z spans the full diameter, X is the near-zero residual). So unlike the
# Matrix-format dataset (where "Ndeg" turned out to be a blade-phase label on one FIXED plane, no
# rotation needed), every angle here needs the rotation below - there is no shortcut "0deg is a
# direct read" case this time.
#
# Rotation convention, confirmed OBJECTIVELY this time (not by eye) - project (x, z) onto a
# standard 2D rotation by the file's own stated angle: r_signed = x*cos(angle) + z*sin(angle),
# thickness = -x*sin(angle) + z*cos(angle). thickness comes out ~0 (mean ~0.0005 m, max exactly
# 0.001 m - the file's own known slice half-thickness) at EVERY angle tested (30/50/130deg) with
# this exact sign - the wrong sign/rotation direction gives thickness ~0.08-0.12 m instead, an
# order of magnitude too large to be the thin probe slice. r_signed is therefore the genuine
# signed radial coordinate (full diameter range, both signs) and thickness is the (near-zero)
# tangential/out-of-plane residual. Velocity rotates the SAME way (it's a vector too): this
# exactly matches tip_vortex_tracking.cylindrical_velocity(..., axis='y') called with
# angle_deg=-angle (the same "negated Ndeg" sign this project's OTHER meridional dataset also
# needed, which is reassuring but still only confirmed for these two datasets, not a general
# rule) - v_r/v_axial come out matching r_signed/y directly; v_theta comes out sign-flipped
# relative to a direct (not through cylindrical_velocity) rotation of (vx, vz) by angle_deg, which
# is fine as an internal convention (an overall tangential-sign choice) as long as it's used
# consistently, which every function below does.

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay, cKDTree


def load_scattered_xyzv(data_file: str, header_lines: int = 15):

    '''
    Parse one PowerVIZ scattered "position x y z value(<Variable>)" .txt
    export - NOT the Matrix format load_meridional_matrix() reads (see
    this section's own header comment for how to tell them apart: check
    the file's own "# Format:" line).

    Returns
    -------
    x, y, z, value : np.ndarray, shape (n_points,), with NaN at any
        "Missing" sample (same convention as load_meridional_matrix()).
    '''

    data = np.genfromtxt(data_file, skip_header=header_lines, filling_values=np.nan)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def rotate_scattered_position(x: np.ndarray, z: np.ndarray, angle_deg: float):

    '''
    (r_signed, thickness) from this plane's own stated angle - see this
    section's own header comment for the sign convention and how it was
    confirmed (thickness ~0 at the correct sign, ~0.1 m - an order of
    magnitude too large for this probe's known half-thickness - at the
    wrong one).

    Returns
    -------
    r_signed, thickness : np.ndarray, same shape as x/z
    '''

    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    r_signed = x * cos_t + z * sin_t
    thickness = -x * sin_t + z * cos_t

    return r_signed, thickness


def _build_scattered_grid_interp(r_signed: np.ndarray, y: np.ndarray, n_r: int, n_y: int,
                                  dist_factor: float = 3.0):

    '''
    Shared setup for griddata_scattered_field()/scattered_cylindrical_components():
    one regular (r_signed, y) grid plus the (triangulation, KDTree,
    distance mask) needed to interpolate any number of value arrays onto
    it consistently and cheaply (build once, reuse per field - avoids
    paying for a fresh Delaunay triangulation per field the way calling
    scipy.interpolate.griddata() separately for each one would).

    Distance mask: grid points farther than `dist_factor` times the
    scattered points' own 99.99th-percentile nearest-neighbor spacing
    from any real sample are excluded (set NaN downstream) - the same
    technique this project's own LSB near-wall analysis used, for the
    same reason: griddata's/LinearNDInterpolator's linear interpolation
    still "succeeds" on the convex hull's outer, genuinely-unsampled
    triangles, producing long, spurious extrapolation artifacts rather
    than the gap in the data that's actually there.

    Returns
    -------
    R, Y : np.ndarray, shape (n_y, n_r), meshgrid
    tri : scipy.spatial.Delaunay
    valid_mask : np.ndarray of bool, shape (n_y, n_r)
    nan_mask_points : np.ndarray of bool, shape (n_points,) - True where
        r_signed/y itself was NaN (a "Missing" sample) - exclude these
        from the triangulation entirely, same as every other point-cloud
        reader in this project.
    '''

    valid_points = ~(np.isnan(r_signed) | np.isnan(y))
    r_valid, y_valid = r_signed[valid_points], y[valid_points]

    r_lin = np.linspace(r_valid.min(), r_valid.max(), n_r)
    y_lin = np.linspace(y_valid.min(), y_valid.max(), n_y)
    R, Y = np.meshgrid(r_lin, y_lin)

    tri = Delaunay(np.column_stack([r_valid, y_valid]))

    tree = cKDTree(np.column_stack([r_valid, y_valid]))
    nn_dist, _ = tree.query(np.column_stack([r_valid, y_valid]), k=2)
    typical_spacing = np.percentile(nn_dist[:, 1], 99.99)
    grid_nn_dist, _ = tree.query(np.column_stack([R.ravel(), Y.ravel()]), k=1)
    valid_mask = (grid_nn_dist <= dist_factor * typical_spacing).reshape(R.shape)

    return R, Y, tri, valid_mask, valid_points


def griddata_scattered_field(data_file: str, angle_deg: float, n_r: int = 300, n_y: int = 150,
                              dist_factor: float = 3.0):

    '''
    Load ONE scattered scalar export (e.g. Lambda2, Static Pressure,
    velocity magnitude - anything that does NOT need cylindrical
    projection, unlike raw X/Y/Z-Velocity - see
    scattered_cylindrical_components() for those) at a given azimuth,
    rotate its position onto (r_signed, y) (see this section's own
    rotation convention), and interpolate onto a regular grid.

    Returns
    -------
    R, Y : np.ndarray, shape (n_y, n_r), meshgrid, meters (r_signed
        already signed/full-diameter - divide by r_tip yourself for a
        plot_meridional_field()-style r/R axis, same as that function's
        own X/Y convention).
    value_grid : np.ndarray, shape (n_y, n_r), NaN outside valid_mask
    '''

    x, y, z, value = load_scattered_xyzv(data_file)
    r_signed, _ = rotate_scattered_position(x, z, angle_deg)

    R, Y, tri, valid_mask, valid_points = _build_scattered_grid_interp(r_signed, y, n_r, n_y, dist_factor)

    interp = LinearNDInterpolator(tri, value[valid_points])
    grid = interp(R, Y)
    nan_mask = np.isnan(grid)
    if np.any(nan_mask):
        nearest = NearestNDInterpolator(np.column_stack([r_signed[valid_points], y[valid_points]]),
                                         value[valid_points])
        grid[nan_mask] = nearest(R[nan_mask], Y[nan_mask])
    grid[~valid_mask] = np.nan

    return R, Y, grid


def scattered_cylindrical_components(x_vel_file: str, y_vel_file: str, z_vel_file: str,
                                      angle_deg: float, n_r: int = 300, n_y: int = 150,
                                      dist_factor: float = 3.0):

    '''
    Load one azimuth's raw X/Y/Z-Velocity SCATTERED exports, project onto
    cylindrical (v_r, v_axial, v_theta) at the SCATTERED-point level
    (see this section's own rotation convention - reuses
    tip_vortex_tracking.cylindrical_velocity(..., axis='y',
    angle_deg=-angle_deg)), then interpolate all three onto ONE shared
    regular (r_signed, y) grid (built once - see
    _build_scattered_grid_interp() - and reused for all three fields,
    not rebuilt per field).

    Returns
    -------
    R, Y : np.ndarray, shape (n_y, n_r), meshgrid, meters
    v_r, v_axial, v_theta : np.ndarray, shape (n_y, n_r), NaN outside
        the shared valid_mask (see _build_scattered_grid_interp())
    '''

    x, y, z, vx = load_scattered_xyzv(x_vel_file)
    _, _, _, vy = load_scattered_xyzv(y_vel_file)
    _, _, _, vz = load_scattered_xyzv(z_vel_file)

    if vy.shape != vx.shape or vz.shape != vx.shape:
        raise ValueError(
            f"x/y/z-velocity files don't share the same point count: "
            f"{vx.shape}, {vy.shape}, {vz.shape}."
        )

    v_r, v_theta, v_axial = cylindrical_velocity(vx, vy, vz, -angle_deg, axis='y')
    r_signed, _ = rotate_scattered_position(x, z, angle_deg)

    R, Y, tri, valid_mask, valid_points = _build_scattered_grid_interp(r_signed, y, n_r, n_y, dist_factor)

    points_valid = np.column_stack([r_signed[valid_points], y[valid_points]])
    grids = []
    for field in (v_r, v_axial, v_theta):
        interp = LinearNDInterpolator(tri, field[valid_points])
        grid = interp(R, Y)
        nan_mask = np.isnan(grid)
        if np.any(nan_mask):
            nearest = NearestNDInterpolator(points_valid, field[valid_points])
            grid[nan_mask] = nearest(R[nan_mask], Y[nan_mask])
        grid[~valid_mask] = np.nan
        grids.append(grid)

    return (R, Y, *grids)
