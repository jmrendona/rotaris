'''
Tip-vortex core tracking and circulation/core-radius fitting, for the
scattered (x, y, z, value) phase-locked meridional-plane sweep (see
meridional_field.py's own "Scattered (x, y, z, value) meridional
exports" section for the file format and this dataset's rotation
convention - reused here directly).

Methodology follows Quaglia, Leonard, Moreau, & Roger (2017), "A 3D
analytical model for orthogonal blade-vortex interaction noise",
J. Sound Vib. 399, 104-123 (10.1016/j.jsv.2017.03.023) - this project's
own PI is a co-author - for the core-tracking/extraction procedure
specifically: a LOCAL, FINER re-interpolated POLAR grid around each
rough core location (smaller spatial extent than the full raw CFD
mesh - that paper's own Fig. 4: "Extraction from the CFD mesh" (the
native, irregular triangulated mesh) vs. "Interpolation mesh" (a clean
regular polar (rho, phi) grid centered on the core, visibly smaller in
radius) - this module's find_vortex_core() does the equivalent
refinement on a local CARTESIAN grid for locating the core itself, and
plot_core_extraction_mesh() below builds the actual POLAR grid, for the
Fig. 4-style visualization specifically), tracked continuously cut to
cut (their own "tracking the vorticity maximum" language), with
agreement between independent core-finding criteria used as a
confidence check ("the standard deviation remains small close to the
vortex core" - see find_vortex_core()'s own 'spread' return value).
Vortex-frame naming (their Fig. 4 axis diagram: r_v radial from the
vortex core, theta_v/v_v azimuthal around it, z_v along the vortex's
OWN axis, u_v the second in-plane Cartesian direction) maps onto this
dataset's own confirmed swirl-plane convention (see this module's next
paragraph) as: their V_theta (swirl, Fig. 5a/6a/7a) = this module's
v_swirl (built from v_r, v_axial here); their V_z (Fig. 5b/6b/7b) =
this module's v_along_core (= this dataset's own v_theta, since the
vortex's own axis is tangential here - see below).

Swirl-plane convention (confirmed empirically for this dataset - see
meridional_field.py): the tip vortex's own local axis is predominantly
TANGENTIAL here (a fixed-azimuth meridional cut sits roughly
perpendicular to it), so its swirl lives in the (v_r, v_axial) plane,
NOT v_theta - confirmed directly: v_r shows the same sharp, localized
dipole at every vortex core that v_axial does, while v_theta shows only
a smooth background field with no comparable core signature. Every
circulation/core-radius quantity below is built from (v_r, v_axial)
alone.
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator
from scipy.spatial import Delaunay
from matplotlib.colors import TwoSlopeNorm

from bladeprocessor.tip_vortex_tracking import cylindrical_velocity
from bladeprocessor.meridional_field import (
    load_scattered_xyzv, rotate_scattered_position, _build_scattered_grid_interp,
    scattered_cylindrical_components, griddata_scattered_field, plot_meridional_field,
)


def rotate_scattered_vector(x: np.ndarray, z: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                             angle_deg: float):

    '''
    (v_r, v_axial, v_theta) for ANY vector field sampled on this
    dataset's own scattered points (velocity OR vorticity - both
    transform the same way under a pure rotation) - thin wrapper around
    tip_vortex_tracking.cylindrical_velocity() with this dataset's own
    confirmed sign convention (angle_deg=-angle_deg, axis='y') - see
    meridional_field.py's own "Scattered (x, y, z, value) meridional
    exports" section for how that sign was confirmed (NOT by eye - the
    spatial rotation that makes the slice's own near-zero thickness
    residual vanish, which must agree with the velocity/vorticity
    rotation since it's the same underlying coordinate rotation).

    Returns
    -------
    v_r, v_axial, v_theta : np.ndarray, same shape as vx/vy/vz
    '''

    v_r, v_theta, v_axial = cylindrical_velocity(vx, vy, vz, -angle_deg, axis='y')
    return v_r, v_axial, v_theta


def find_vortex_core(pl_master_path: str, angle_deg: float, r_center: float, y_center: float,
                      search_radius: float = 0.02, refine_n: int = 81):

    '''
    Refine a rough tip-vortex core location into a sub-grid-accurate one
    using a SMALL, FINE local re-interpolation (Quaglia et al.'s own
    "interpolated grid has a smaller radius... than the raw grid") and
    THREE independent criteria over it, so their mutual agreement is a
    direct confidence check (their "standard deviation remains small
    close to the vortex core"):

    - lambda2 minimum (Jeong & Hussain 1995 vortex-core criterion)
    - static pressure minimum (a vortex core is a local pressure
      minimum - the simpler, classic criterion, more robust to mesh/
      interpolation noise than any derivative-based one)
    - |omega_theta| maximum (vorticity-component peak, matching Quaglia
      et al.'s own stated criterion directly - omega_theta since the
      vortex axis is tangential here, see this module's own docstring)

    Parameters
    ----------
    pl_master_path : str
        e.g. '/scratch/jmrendon/Rotor-alone/6e-5_6000rpm_PL'
    angle_deg : float
    r_center, y_center : float
        Rough center to refine around (e.g. the previous angle's own
        refined core location, continuity-tracked cut to cut - see this
        module's own track_vortex_core_sweep()).
    search_radius : float
        Half-width [m] of the square local window to re-interpolate
        within - must comfortably contain the core but stay much
        smaller than the full field (hence "smaller radius... than the
        raw grid") - tune against how far the rough estimate can be off
        (continuity tracking keeps this small cut-to-cut; widen for the
        FIRST cut's own bootstrap, which has no previous cut to start
        from).
    refine_n : int
        Local fine-grid resolution (refine_n x refine_n) - odd, so the
        grid has an exact center sample.

    Returns
    -------
    dict with:
        'r_l2', 'y_l2'     : lambda2-minimum location
        'r_p', 'y_p'       : pressure-minimum location
        'r_vor', 'y_vor'   : |omega_theta|-maximum location
        'r_core', 'y_core' : mean of the three (the core location to use downstream)
        'spread'           : max pairwise distance [m] between the three - LARGE means
                             the criteria disagree at this cut (flag it, don't trust blindly)
        'l2_min', 'p_min', 'omega_theta_max' : the extremal values themselves
    '''

    angle_str = f'{int(round(angle_deg)):02d}'
    l2_x, l2_y, l2_z, l2_val = load_scattered_xyzv(f'{pl_master_path}/data/l2/lambda2_{angle_str}deg_avg.txt')
    _, _, _, p_val = load_scattered_xyzv(f'{pl_master_path}/data/pstat/p_stat_{angle_str}deg_avg.txt')
    _, _, _, vor_x = load_scattered_xyzv(f'{pl_master_path}/data/x-vor/vort_x_{angle_str}deg_avg.txt')
    _, _, _, vor_y = load_scattered_xyzv(f'{pl_master_path}/data/y-vor/vort_y_{angle_str}deg_avg.txt')
    _, _, _, vor_z = load_scattered_xyzv(f'{pl_master_path}/data/z-vor/vort_z_{angle_str}deg_avg.txt')

    r_signed, _ = rotate_scattered_position(l2_x, l2_z, angle_deg)
    _, _, omega_theta = rotate_scattered_vector(l2_x, l2_z, vor_x, vor_y, vor_z, angle_deg)

    window = (
        (r_signed >= r_center - search_radius) & (r_signed <= r_center + search_radius) &
        (l2_y >= y_center - search_radius) & (l2_y <= y_center + search_radius) &
        ~np.isnan(r_signed) & ~np.isnan(l2_y)
    )
    if window.sum() < 10:
        raise ValueError(
            f"Only {int(window.sum())} scattered points in the search window at angle={angle_deg} - "
            "widen search_radius or check r_center/y_center."
        )

    r_w, y_w = r_signed[window], l2_y[window]
    r_lin = np.linspace(r_center - search_radius, r_center + search_radius, refine_n)
    y_lin = np.linspace(y_center - search_radius, y_center + search_radius, refine_n)
    R_fine, Y_fine = np.meshgrid(r_lin, y_lin)

    def _interp_local(values):
        interp = LinearNDInterpolator(np.column_stack([r_w, y_w]), values[window])
        grid = interp(R_fine, Y_fine)
        if np.any(np.isnan(grid)):
            nearest = NearestNDInterpolator(np.column_stack([r_w, y_w]), values[window])
            nan_mask = np.isnan(grid)
            grid[nan_mask] = nearest(R_fine[nan_mask], Y_fine[nan_mask])
        return grid

    l2_grid = _interp_local(l2_val)
    p_grid = _interp_local(p_val)
    omega_theta_grid = _interp_local(omega_theta)

    def _argext(grid, mode):
        idx = np.argmin(grid) if mode == 'min' else np.argmax(np.abs(grid))
        iy, ix = np.unravel_index(idx, grid.shape)
        return r_lin[ix], y_lin[iy], grid[iy, ix]

    r_l2, y_l2, l2_min = _argext(l2_grid, 'min')
    r_p, y_p, p_min = _argext(p_grid, 'min')
    r_vor, y_vor, omega_theta_max = _argext(omega_theta_grid, 'absmax')

    r_core = float(np.mean([r_l2, r_p, r_vor]))
    y_core = float(np.mean([y_l2, y_p, y_vor]))
    candidates = np.array([[r_l2, y_l2], [r_p, y_p], [r_vor, y_vor]])
    spread = float(np.max(np.linalg.norm(candidates[:, None, :] - candidates[None, :, :], axis=-1)))

    return {
        'r_l2': r_l2, 'y_l2': y_l2, 'l2_min': float(l2_min),
        'r_p': r_p, 'y_p': y_p, 'p_min': float(p_min),
        'r_vor': r_vor, 'y_vor': y_vor, 'omega_theta_max': float(omega_theta_max),
        'r_core': r_core, 'y_core': y_core, 'spread': spread,
    }


def azimuthal_profiles(R: np.ndarray, Y: np.ndarray, v_r: np.ndarray, v_axial: np.ndarray,
                        v_theta: np.ndarray, r_core: float, y_core: float, rho_values: np.ndarray,
                        n_phi: int = 72):

    '''
    BOTH azimuthally-averaged profiles around (r_core, y_core) needed to
    match Quaglia et al.'s own Fig. 5/6/7 pair (vortex-frame V_theta and
    V_z - see this module's own docstring for the exact mapping):

    - v_swirl(rho): swirl velocity IN THE (r, y) PLANE (not around the
      rotor's own axis) - the standard rotor-wake-PIV construction (e.g.
      Ramasamy & Leishman), built from (v_r, v_axial) since that's this
      dataset's own confirmed swirl plane - matches their V_theta.
    - v_along_core(rho): plain azimuthal average of v_theta itself (this
      dataset's rotor-frame tangential component) at fixed rho - no
      circle-tangent projection needed, it's already the component ALONG
      the vortex's own axis here - matches their V_z.

    At each rho in rho_values, samples n_phi points equally spaced around
    the circle of radius rho centered on the core, bilinearly
    interpolates (v_r, v_axial, v_theta) there (RegularGridInterpolator,
    since R/Y are already a regular meshgrid), and averages over phi
    (v_swirl additionally projects (v_r, v_axial) onto the circle's own
    local tangential direction (-sin(phi), cos(phi)) first).

    Returns
    -------
    v_swirl_mean, v_swirl_std, v_along_core_mean, v_along_core_std :
        np.ndarray, shape (len(rho_values),) each - NaN where a sample
        point fell outside R/Y's own valid (non-NaN) extent. The *_std
        values are std ACROSS PHI at fixed rho, i.e. how far from
        axisymmetric the real (steady, phase-locked) field is there, not
        an unsteadiness/noise measure.
    '''

    r_axis = R[0, :]
    y_axis = Y[:, 0]
    interp_r = RegularGridInterpolator((y_axis, r_axis), v_r, bounds_error=False, fill_value=np.nan)
    interp_a = RegularGridInterpolator((y_axis, r_axis), v_axial, bounds_error=False, fill_value=np.nan)
    interp_t = RegularGridInterpolator((y_axis, r_axis), v_theta, bounds_error=False, fill_value=np.nan)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)

    v_swirl_mean = np.full(len(rho_values), np.nan)
    v_swirl_std = np.full(len(rho_values), np.nan)
    v_along_core_mean = np.full(len(rho_values), np.nan)
    v_along_core_std = np.full(len(rho_values), np.nan)

    for i, rho in enumerate(rho_values):
        r_samples = r_core + rho * cos_phi
        y_samples = y_core + rho * sin_phi
        pts = np.column_stack([y_samples, r_samples])
        vr_samples = interp_r(pts)
        va_samples = interp_a(pts)
        vt_samples = interp_t(pts)

        v_swirl = -vr_samples * sin_phi + va_samples * cos_phi
        if not np.all(np.isnan(v_swirl)):
            v_swirl_mean[i] = np.nanmean(v_swirl)
            v_swirl_std[i] = np.nanstd(v_swirl)
        if not np.all(np.isnan(vt_samples)):
            v_along_core_mean[i] = np.nanmean(vt_samples)
            v_along_core_std[i] = np.nanstd(vt_samples)

    return v_swirl_mean, v_swirl_std, v_along_core_mean, v_along_core_std


def azimuthal_scalar_profile(R: np.ndarray, Y: np.ndarray, scalar: np.ndarray, r_core: float,
                              y_core: float, rho_values: np.ndarray, n_phi: int = 72):

    '''
    Plain azimuthal average of ANY already-gridded scalar field (e.g.
    static pressure from griddata_scattered_field()) around
    (r_core, y_core) - the same sampling-circle construction as
    azimuthal_profiles()'s own v_along_core, generalized to any scalar
    so it doesn't need its own velocity-rotation treatment.

    Returns
    -------
    mean, std : np.ndarray, shape (len(rho_values),)
    '''

    r_axis = R[0, :]
    y_axis = Y[:, 0]
    interp = RegularGridInterpolator((y_axis, r_axis), scalar, bounds_error=False, fill_value=np.nan)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    mean = np.full(len(rho_values), np.nan)
    std = np.full(len(rho_values), np.nan)

    for i, rho in enumerate(rho_values):
        r_samples = r_core + rho * np.cos(phi)
        y_samples = y_core + rho * np.sin(phi)
        samples = interp(np.column_stack([y_samples, r_samples]))
        if not np.all(np.isnan(samples)):
            mean[i] = np.nanmean(samples)
            std[i] = np.nanstd(samples)

    return mean, std


def lamb_oseen(rho, gamma, r_c):
    '''v_theta(rho) = (Gamma / 2*pi*rho) * (1 - exp(-1.25643*rho**2/r_c**2)) - the 1.25643 factor
    (ln 2 doubled, i.e. alpha=1.25643) is the standard convention making r_c the radius of PEAK
    swirl velocity exactly (not just an arbitrary viscous diffusion lengthscale).'''
    rho = np.asarray(rho, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = (gamma / (2 * np.pi * rho)) * (1 - np.exp(-1.25643 * rho ** 2 / r_c ** 2))
    return np.where(rho == 0, 0.0, v)


def scully(rho, gamma, r_c):
    '''v_theta(rho) = Gamma*rho / (2*pi*(rho**2 + r_c**2)) - Kaufmann/Scully (1975) algebraic
    model - the first-term Taylor-expansion simplification of the Lamb-Oseen exponential
    (Bhagwat & Leishman), later generalized by Vatistas as the n=1 case of vatistas_n2's own
    family (n=2 there, NOT the same model - kept as two separate, explicitly-named functions
    to match the literature's own naming, e.g. Quaglia et al.'s Fig. 7).'''
    rho = np.asarray(rho, dtype=float)
    return gamma * rho / (2 * np.pi * (rho ** 2 + r_c ** 2))


def vatistas_n2(rho, gamma, r_c):
    '''v_theta(rho) = Gamma*rho / (2*pi*(r_c**4 + rho**4)**0.5) - Vatistas (1991) algebraic model,
    n=2 (a common rotorcraft/BVI-literature choice - n=1 is the Scully model, n->inf is Rankine).'''
    rho = np.asarray(rho, dtype=float)
    return gamma * rho / (2 * np.pi * np.sqrt(r_c ** 4 + rho ** 4))


def taylor(rho, G, r_c):
    '''v_theta(rho) = G * (rho/r_c) * exp(-0.5*(rho/r_c)**2) - the "Taylor vortex" model used in
    several BVI papers (e.g. Quaglia et al.'s own Fig. 7a) - a COMPACT model: unlike
    lamb_oseen/scully/vatistas_n2, which all asymptote to Gamma/(2*pi*rho) far from the core, this
    decays to ZERO at large rho (known in the literature to decay faster / underpredict the
    far-field swirl compared to real tip vortices - kept for the same direct comparison Quaglia et
    al. make, not because it's expected to win). G is a strength parameter, NOT directly the same
    "total circulation at infinity" that Gamma means for the other three models (this model has no
    such asymptote) - don't compare G and Gamma values directly across models. Peak velocity is at
    rho=r_c (same convention as the other three), confirmed by d/drho=0 there.'''
    rho = np.asarray(rho, dtype=float)
    return G * (rho / r_c) * np.exp(-0.5 * (rho / r_c) ** 2)


def fit_vortex_model(rho_values: np.ndarray, v_profile_mean: np.ndarray, model=lamb_oseen):

    '''
    Fit `model` (lamb_oseen, scully, vatistas_n2, or taylor - all take
    (rho, strength, r_c)) to an azimuthal_profiles() v_swirl_mean result
    via nonlinear least squares, using the profile's own peak as the
    initial guess (robust - every model's peak location/value is close
    to (r_c, strength/(2*pi*r_c)*0.7-ish) for a reasonable initial
    guess, and curve_fit converges reliably from there in practice on
    this kind of smooth, single-humped profile).

    Returns
    -------
    strength, r_c : float - fitted Gamma [m^2/s] (or G [m/s] for taylor
        - see that function's own docstring) and core radius [m]
    pcov : np.ndarray, shape (2, 2) - covariance, for a quick quality
        check (large diagonal entries relative to strength**2/r_c**2
        mean a poor/ambiguous fit, e.g. too few valid rho samples).
    '''

    valid = ~np.isnan(v_profile_mean)
    rho_fit, v_fit = rho_values[valid], v_profile_mean[valid]

    i_peak = np.argmax(np.abs(v_fit))
    r_c0 = max(rho_fit[i_peak], 1e-4)
    strength0 = 2 * np.pi * r_c0 * v_fit[i_peak] / 0.715  # ~0.715 = LO's own peak-swirl-velocity factor

    (strength, r_c), pcov = curve_fit(model, rho_fit, v_fit, p0=[strength0, r_c0], maxfev=5000)
    return strength, r_c, pcov


def viscous_peak_swirl_decay(t, amplitude, t0):

    '''
    v_max(t) = amplitude / sqrt(t + t0) - the long-time peak-swirl-velocity decay implied by
    classical (Lamb-Oseen) viscous diffusion of a vortex core: the core radius grows as
    r_c(t) = sqrt(r_c0^2 + 4*alpha*nu_eff*t) (alpha=1.25643, nu_eff an EFFECTIVE/eddy viscosity -
    real tip vortices diffuse far faster than molecular nu would predict, so this is never meant
    to recover a literal viscosity), and for roughly-conserved circulation Gamma,
    v_max(t) = Gamma/(2*pi*r_c(t)) * (1-exp(-alpha)) ~ 1/r_c(t) ~ (t+t0)^-0.5, with `t0` absorbing
    r_c0^2/(4*alpha*nu_eff) and `amplitude` absorbing Gamma*(1-exp(-alpha))/(2*pi*sqrt(4*alpha*nu_eff)).
    Neither fitted parameter is separable into Gamma/r_c0/nu_eff from this alone - this checks the
    FUNCTIONAL FORM (does the measured decay actually go as (time)^-1/2?), not a literal viscosity.
    '''

    t = np.asarray(t, dtype=float)
    return amplitude / np.sqrt(t + t0)


def fit_viscous_peak_swirl_decay(t_values: np.ndarray, v_peak_values: np.ndarray):

    '''
    curve_fit wrapper for viscous_peak_swirl_decay() against a measured peak-swirl-velocity vs
    time series (e.g. v_swirl_max from tip_vortex_manager.py's own per-wake-age tracking loop,
    with wake age converted to a time via the rotor's RPM).

    Returns
    -------
    amplitude, t0 : float - fitted parameters (see viscous_peak_swirl_decay()'s own docstring -
        not literally Gamma/r_c0/nu_eff individually)
    r_squared : float - coefficient of determination against the data, a quick "does this
        functional form actually describe the decay" check (not a substitute for looking at the
        plot - a high R^2 over a short/noisy wake-age range can still be misleading).
    '''

    t_values = np.asarray(t_values, dtype=float)
    v_peak_values = np.asarray(v_peak_values, dtype=float)

    dt_sorted = np.diff(np.sort(t_values))
    dt_sorted = dt_sorted[dt_sorted > 0]
    t0_guess = max(float(dt_sorted.min()), 1e-8) if dt_sorted.size else 1e-4
    amp_guess = v_peak_values[0] * np.sqrt(max(t_values[0] + t0_guess, t0_guess))

    popt, _ = curve_fit(viscous_peak_swirl_decay, t_values, v_peak_values,
                         p0=[amp_guess, t0_guess], bounds=([0.0, 1e-8], [np.inf, np.inf]), maxfev=5000)
    v_fit = viscous_peak_swirl_decay(t_values, *popt)
    ss_res = np.sum((v_peak_values - v_fit) ** 2)
    ss_tot = np.sum((v_peak_values - np.mean(v_peak_values)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return float(popt[0]), float(popt[1]), r_squared


def plot_core_extraction_mesh(pl_master_path: str, angle_deg: float, r_core: float, y_core: float,
                               field_folder: str, field_prefix: str, variable_label: str,
                               window: float = 0.012, n_rho: int = 60, n_phi: int = 72,
                               cmap: str = 'RdBu_r', saturation: float = 0.6,
                               savepath_mesh: str = None, savepath_polar: str = None):

    '''
    Quaglia et al.'s own Fig. 4 pair, reproduced as two SEPARATE images:
    (a) the scalar field on the RAW scattered CFD points, local to the
    vortex core, with the actual Delaunay triangulation overlaid (their
    "Extraction from the CFD mesh" - the irregular, direction-dependent
    native mesh density is the whole point of showing this one); (b) the
    SAME field re-interpolated onto a clean, regular POLAR (rho, phi)
    grid centered on (r_core, y_core) (their "Interpolation mesh") - the
    grid this module's azimuthal_profiles()/fit_vortex_model() actually
    sample on, made visible.

    Parameters
    ----------
    field_folder, field_prefix : str
        e.g. 'pstat', 'p_stat' for
        data/pstat/p_stat_<N>deg_avg.txt (matches this dataset's own
        per-variable folder/filename convention - see
        meridional_field.py's own scattered-format functions).
    window : float
        Half-width [m] of the square LOCAL region (around the core)
        shown in panel (a) and spanned by panel (b)'s own outer radius -
        keep this small relative to the full field, same spirit as
        find_vortex_core()'s own search_radius (Quaglia et al.'s own
        "interpolation mesh has a smaller radius... than the raw grid").
    n_rho, n_phi : int
        Polar grid resolution for panel (b).
    cmap : str
        Diverging colormap, default 'RdBu_r' (red = above local ambient,
        blue = below) - makes the low-pressure vortex core visually pop
        against the ambient field, unlike a sequential/gray map.
    saturation : float
        Fraction (0-1) of the local raw-data's 2nd-98th percentile spread
        (about its median, taken as the local "ambient" level) actually
        used for the colour limits - lower values clip more of the range
        into the strongest colour, boosting contrast at the core. Both
        panels share the SAME limits (from the raw data in panel (a)) so
        they stay directly comparable.

    Returns
    -------
    (fig_mesh, ax_mesh), (fig_polar, ax_polar)
    '''

    angle_str = f'{int(round(angle_deg)):02d}'
    x, y, z, value = load_scattered_xyzv(
        f'{pl_master_path}/data/{field_folder}/{field_prefix}_{angle_str}deg_avg.txt')
    r_signed, _ = rotate_scattered_position(x, z, angle_deg)

    local = (
        (r_signed >= r_core - window) & (r_signed <= r_core + window) &
        (y >= y_core - window) & (y <= y_core + window) &
        ~np.isnan(r_signed) & ~np.isnan(y) & ~np.isnan(value)
    )
    r_loc, y_loc, v_loc = r_signed[local], y[local], value[local]

    # Shared, saturated diverging norm (centered on the local "ambient" level, taken as the
    # median of the raw local data) - used for BOTH panels so they stay comparable, and clipped
    # to `saturation` x the 2nd-98th percentile spread so the vortex's own pressure dip actually
    # saturates the colormap instead of being washed out by the full (often outlier-driven) range.
    center = float(np.nanmedian(v_loc))
    p_lo, p_hi = np.nanpercentile(v_loc, [2, 98])
    half_range = max(center - p_lo, p_hi - center) * saturation
    half_range = half_range if half_range > 0 else 1.0
    norm = TwoSlopeNorm(vmin=center - half_range, vcenter=center, vmax=center + half_range)

    # --- (a) raw CFD mesh ---
    fig_mesh, ax_mesh = plt.subplots(figsize=(6.5, 6))
    tri = Delaunay(np.column_stack([r_loc, y_loc]))
    tpc = ax_mesh.tripcolor(r_loc * 1000, y_loc * 1000, tri.simplices, v_loc, cmap=cmap, norm=norm, shading='flat')
    ax_mesh.triplot(r_loc * 1000, y_loc * 1000, tri.simplices, color='k', linewidth=0.15, alpha=0.4)
    ax_mesh.plot(r_core * 1000, y_core * 1000, '+', color='red', markersize=14, markeredgewidth=2)
    cbar = fig_mesh.colorbar(tpc, ax=ax_mesh)
    cbar.set_label(variable_label)
    ax_mesh.set_xlabel(r'$r$ [mm] (local, core-centered)')
    ax_mesh.set_ylabel(r'$y$ [mm] (local, core-centered)')
    ax_mesh.set_aspect('equal')
    fig_mesh.tight_layout()
    if savepath_mesh:
        fig_mesh.savefig(savepath_mesh, dpi=300)

    # --- (b) polar interpolation mesh ---
    rho_lin = np.linspace(0, window, n_rho)
    phi_lin = np.linspace(0, 2 * np.pi, n_phi)
    RHO, PHI = np.meshgrid(rho_lin, phi_lin)
    R_polar = r_core + RHO * np.cos(PHI)
    Y_polar = y_core + RHO * np.sin(PHI)

    interp = LinearNDInterpolator(np.column_stack([r_loc, y_loc]), v_loc)
    V_polar = interp(R_polar, Y_polar)
    nan_mask = np.isnan(V_polar)
    if np.any(nan_mask):
        nearest = NearestNDInterpolator(np.column_stack([r_loc, y_loc]), v_loc)
        V_polar[nan_mask] = nearest(R_polar[nan_mask], Y_polar[nan_mask])

    fig_polar, ax_polar = plt.subplots(figsize=(6.5, 6))
    pcm = ax_polar.pcolormesh((R_polar - r_core) * 1000, (Y_polar - y_core) * 1000, V_polar,
                               cmap=cmap, norm=norm, shading='gouraud')
    for rho in rho_lin[::max(1, n_rho // 10)]:
        circle = plt.Circle((0, 0), rho * 1000, fill=False, color='k', linewidth=0.3, alpha=0.6)
        ax_polar.add_patch(circle)
    for phi in phi_lin[::max(1, n_phi // 24)]:
        ax_polar.plot([0, window * 1000 * np.cos(phi)], [0, window * 1000 * np.sin(phi)],
                      color='k', linewidth=0.3, alpha=0.6)
    ax_polar.plot(0, 0, '+', color='red', markersize=14, markeredgewidth=2)
    cbar = fig_polar.colorbar(pcm, ax=ax_polar)
    cbar.set_label(variable_label)
    ax_polar.set_xlabel(r'$r - r_{\rm core}$ [mm]')
    ax_polar.set_ylabel(r'$y - y_{\rm core}$ [mm]')
    ax_polar.set_aspect('equal')
    fig_polar.tight_layout()
    if savepath_polar:
        fig_polar.savefig(savepath_polar, dpi=300)

    return (fig_mesh, ax_mesh), (fig_polar, ax_polar)


# ------------------------------------------------------------------------------------------------
# Full wake-age sweep: tracking loop + every downstream diagnostic/figure, each as its own
# independently-callable function (so tip_vortex_manager.py can stay a thin, commentable sequence
# of calls, matching this project's other *_manager.py files - the computation/plotting itself
# lives here, not inline in the manager). All of these consume `track_results` (the list of dicts
# returned by track_vortex_core_sweep() below, one per tracked angle, each already containing
# find_vortex_core()'s own output merged with 'angle', every model's fitted strength_<name>/
# rc_<name>, 'rc_peak'/'strength_peak' (data-only, see plot_circulation_vs_wakeage()'s own
# docstring), and 'v_swirl_max') and/or `field_cache` (the dict of per-tracked-angle gridded
# velocity fields + core dict, also returned by track_vortex_core_sweep() - kept in memory across
# the whole sweep rather than reloaded from disk, since several figures below reuse it).
# ------------------------------------------------------------------------------------------------


def wake_age_series(track_results):
    '''
    Wake age [deg] for every row in `track_results`, defined relative to track_results[0]'s OWN
    angle (the bootstrap/first-tracked angle, i.e. t=0) - NOT assumed to be any fixed number, so
    this works regardless of which angle the sweep actually started from (see
    track_vortex_core_sweep()'s own docstring for why the sweep direction/starting angle matters).
    '''
    bootstrap_angle = track_results[0]['angle']
    return np.array([bootstrap_angle - row['angle'] for row in track_results], dtype=float)


def angle_at_wake_age(track_results, target_wake_age):
    '''The tracked angle (a valid field_cache/track_results key) whose wake age is closest to
    `target_wake_age` - e.g. angle_at_wake_age(track_results, 160) for Quaglia et al.'s own
    "85%"-style late/most-developed wake age used in several figures below.'''
    wake_age = wake_age_series(track_results)
    idx = int(np.argmin(np.abs(wake_age - target_wake_age)))
    return track_results[idx]['angle']


def plot_swirl_profile_fit(rho_values, v_swirl_mean, v_swirl_std, row, models, model_labels,
                            model_styles, savepath):
    '''Per-angle sanity plot: the measured azimuthally-averaged swirl profile (with its azimuthal
    std band) against every fitted vortex model in `models` (row must already carry that model's
    own strength_<name>/rc_<name> - see fit_vortex_model()).'''
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rho_values * 1000, v_swirl_mean, 'o', color='k', label='data (azim. mean)', markersize=4)
    ax.fill_between(rho_values * 1000, v_swirl_mean - v_swirl_std, v_swirl_mean + v_swirl_std,
                     alpha=0.15, color='k')
    rho_fine = np.linspace(rho_values.min(), rho_values.max(), 200)
    for model_name, model in models.items():
        ax.plot(rho_fine * 1000, model(rho_fine, row[f'strength_{model_name}'], row[f'rc_{model_name}']),
                 model_styles[model_name], label=model_labels[model_name])
    ax.set_xlabel(r'$\rho$ [mm]')
    ax.set_ylabel(r'$v_{\rm swirl}$ [m/s]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def track_vortex_core_sweep(pl_master_path, track_angles, r_tip, rho_values, models, model_labels,
                             model_styles, bootstrap_angle=None, search_radius=0.008,
                             bootstrap_search_radius=0.035, refine_n=121,
                             profile_plot_dir=None, results_savepath=None):

    '''
    Track the tip-vortex core continuously across `track_angles` (continuity: each angle's own
    rough center is the PREVIOUS angle's refined core location - see find_vortex_core()), fitting
    every model in `models` to the swirl profile at each cut.

    IMPORTANT: `track_angles` should be ordered so the FIRST entry is wake age 0 (t=0) - NOT
    necessarily ascending. For this dataset, every single angle file already shows MULTIPLE
    discrete vortex cores at once (~0.2R apart in y/R - 1st/2nd/3rd+ blade-passage-old vortices,
    all visible simultaneously in this phase-locked average), and sweeping in the "wrong" order
    (confirmed directly: plain ascending angle, 0->170deg) produced a non-monotonic/backward-
    moving trajectory - the continuity tracker jumping between adjacent "rungs" of that comb
    rather than following one. The working order found by direct inspection is DESCENDING angle
    (e.g. 170, 160, ..., 0), i.e. `bootstrap_angle` is whichever angle gave a clean, monotonic
    trajectory - track_angles[0] by default. `search_radius` is tightened accordingly (well under
    half that ~0.2R spacing) for every step except the first, which has no previous cut to start
    from and uses `bootstrap_search_radius` instead.

    Parameters
    ----------
    pl_master_path : str
    track_angles : list of int/float - sweep order, see above.
    r_tip : float - only used to scale the bootstrap search window (0.6-1.05 R radially,
        -0.3-0.05 R axially - the tip-vortex region for this rotor/dataset).
    rho_values : np.ndarray - sample radii for azimuthal_profiles()/fit_vortex_model().
    models, model_labels, model_styles : dict, keyed by model name (e.g. 'lo', 'scully', ...) -
        models maps to the callables themselves (lamb_oseen, scully, ...).
    bootstrap_angle : float, optional - defaults to track_angles[0].
    profile_plot_dir : str, optional - if given, saves a plot_swirl_profile_fit() sanity plot per
        angle as swirl_profile_<NN>deg.png there.
    results_savepath : str, optional - if given, saves track_results (+ rho_values + track_angles)
        as an .npz there (reload with np.load(..., allow_pickle=True)['results']).

    Returns
    -------
    track_results : list of dict, one per angle in `track_angles`, each already containing
        find_vortex_core()'s own keys plus 'angle', 'strength_<model>'/'rc_<model>'/'fit_ok_<model>'
        for every model, 'rc_peak'/'strength_peak' (data-only, no profile shape assumed), and
        'v_swirl_max'.
    field_cache : dict keyed by angle - {'R', 'Y', 'v_r', 'v_a', 'v_t', 'core'} per angle, kept in
        memory for the figures below that need the full gridded field again (not just
        track_results' own scalar summaries).
    '''

    bootstrap_angle = track_angles[0] if bootstrap_angle is None else bootstrap_angle
    bootstrap_str = f'{int(round(bootstrap_angle)):02d}'
    print(f"Bootstrapping core location at {bootstrap_angle:g}deg (t=0)...")
    x0, y0, z0, l2_0 = load_scattered_xyzv(f'{pl_master_path}/data/l2/lambda2_{bootstrap_str}deg_avg.txt')
    r0, _ = rotate_scattered_position(x0, z0, bootstrap_angle)
    bootstrap_window = (
        (r0 > 0.6 * r_tip) & (r0 < 1.05 * r_tip) & (y0 > -0.3 * r_tip) & (y0 < 0.05 * r_tip)
    )
    i_min = np.nanargmin(np.where(bootstrap_window, l2_0, np.nan))
    r_center, y_center = float(r0[i_min]), float(y0[i_min])
    print(f"  bootstrap core: r={r_center:.5f} y={y_center:.5f}")

    track_results = []
    field_cache = {}
    first = True

    for angle in track_angles:
        sr = bootstrap_search_radius if first else search_radius
        core = find_vortex_core(pl_master_path, float(angle), r_center, y_center, search_radius=sr,
                                 refine_n=refine_n)
        first = False
        r_center, y_center = core['r_core'], core['y_core']

        angle_str = f'{int(round(angle)):02d}'
        x_file = f'{pl_master_path}/data/x-vel/vel_x_{angle_str}deg_avg.txt'
        y_file = f'{pl_master_path}/data/y-vel/vel_y_{angle_str}deg_avg.txt'
        z_file = f'{pl_master_path}/data/z-vel/vel_z_{angle_str}deg_avg.txt'
        R, Y, v_r, v_a, v_t = scattered_cylindrical_components(
            x_file, y_file, z_file, angle_deg=float(angle), n_r=250, n_y=125,
        )
        field_cache[angle] = dict(R=R, Y=Y, v_r=v_r, v_a=v_a, v_t=v_t, core=core)

        v_swirl_mean, v_swirl_std, v_along_mean, v_along_std = azimuthal_profiles(
            R, Y, v_r, v_a, v_t, core['r_core'], core['y_core'], rho_values,
        )

        row = {'angle': angle, **core}
        for model_name, model in models.items():
            strength, r_c, pcov = fit_vortex_model(rho_values, v_swirl_mean, model=model)
            row[f'strength_{model_name}'] = strength
            row[f'rc_{model_name}'] = r_c
            row[f'fit_ok_{model_name}'] = True

        valid = ~np.isnan(v_swirl_mean)
        i_peak = np.argmax(np.abs(v_swirl_mean[valid]))
        rho_valid = rho_values[valid]
        row['rc_peak'] = float(rho_valid[i_peak])
        row['strength_peak'] = float(2 * np.pi * rho_valid[i_peak] * v_swirl_mean[valid][i_peak])
        row['v_swirl_max'] = float(np.nanmax(np.abs(v_swirl_mean)))
        track_results.append(row)
        print(f"angle={angle}: core=({core['r_core']/r_tip:.3f}R,{core['y_core']/r_tip:.3f}R) "
              f"spread={core['spread']*1000:.3f}mm  rc_lo={row['rc_lo']*1000:.2f}mm")

        if profile_plot_dir:
            plot_swirl_profile_fit(
                rho_values, v_swirl_mean, v_swirl_std, row, models, model_labels, model_styles,
                savepath=f'{profile_plot_dir}/swirl_profile_{angle_str}deg.png',
            )

    if results_savepath:
        np.savez(results_savepath, results=track_results, rho_values=rho_values, angles=track_angles)

    return track_results, field_cache


def plot_core_trajectory(track_results, r_tip, savepath):
    '''Core trajectory (r_core/R, y_core/R), colored by wake age - radial contraction/axial
    convection at a glance. No title/angle labels by default (kept minimal per feedback - add
    them back locally if useful for a specific figure, not by default).'''
    wake_age = wake_age_series(track_results)
    r_core_arr = np.array([row['r_core'] for row in track_results]) / r_tip
    y_core_arr = np.array([row['y_core'] for row in track_results]) / r_tip
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(r_core_arr, y_core_arr, c=wake_age, cmap='viridis', s=60, zorder=3)
    ax.plot(r_core_arr, y_core_arr, '-', color='gray', alpha=0.6, zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'wake age $\psi_w$ [deg]')
    ax.set_xlabel('$r_{\\rm core}/R$')
    ax.set_ylabel('$y_{\\rm core}/R$')
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_circulation_vs_wakeage(track_results, models, model_labels, model_styles, savepath,
                                 exclude_models=('taylor',)):
    '''
    Gamma_v vs wake age. Each curve is a least-squares fit (fit_vortex_model) of that model's
    profile SHAPE to the ACTUAL measured swirl data at each wake age - i.e. each is a circulation
    estimate extracted from the real data, just read through a different assumed core shape
    (cross-validation between methods, not model vs ground truth). `exclude_models` leaves out
    models whose own strength parameter isn't a true circulation on the same scale as the others
    (Taylor's G, by default - see taylor()'s own docstring: G~20 vs Gamma~0.3 for the rest, so
    plotting it alongside makes the others look like flat zero). A black dashed "data only" line
    (Gamma_peak = 2*pi*rho_peak*v_peak, from the measured peak location/velocity alone, no model
    shape imposed at all) is always included as the closest thing to a raw-data anchor.
    '''
    wake_age = wake_age_series(track_results)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for model_name in models:
        if model_name in exclude_models:
            continue
        ax.plot(wake_age, [row[f'strength_{model_name}'] for row in track_results],
                model_styles[model_name], marker='o', markersize=5, label=model_labels[model_name])
    ax.plot(wake_age, [row['strength_peak'] for row in track_results],
            'k--', marker='x', markersize=6, label=r'data only ($2\pi\rho_{\rm peak}v_{\rm peak}$)')
    ax.set_xlabel(r'wake age $\psi_w$ [deg]')
    ax.set_ylabel(r'$\Gamma_v$ [m$^2$/s]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_coreradius_vs_wakeage(track_results, models, model_labels, model_styles, savepath):
    '''
    r_c vs wake age for every model (same already-data-derived nature as circulation - see
    plot_circulation_vs_wakeage()'s own docstring), plus the data-only rho_peak reference. Unlike
    circulation, r_c IS directly comparable across all models (same physical meaning regardless
    of assumed shape), so nothing is excluded here.
    '''
    wake_age = wake_age_series(track_results)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_name in models:
        ax.plot(wake_age, [row[f'rc_{model_name}'] * 1000 for row in track_results],
                model_styles[model_name], marker='o', markersize=4, label=model_labels[model_name])
    ax.plot(wake_age, [row['rc_peak'] * 1000 for row in track_results],
            'k--', marker='x', markersize=6, label=r'data only ($\rho_{\rm peak}$)')
    ax.set_xlabel(r'wake age $\psi_w$ [deg]')
    ax.set_ylabel(r'$r_c$ [mm]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_core_trajectory_overlay(track_results, field_cache, overlay_angle, r_tip, savepath):
    '''Core trajectory (same data as plot_core_trajectory()) overlaid directly on one
    representative meridional Ua field (field_cache[overlay_angle], as returned by
    track_vortex_core_sweep() - no extra load/interpolation needed) - shows the vortex path
    against the actual mean flow it sits in, rather than a blank background.'''
    wake_age = wake_age_series(track_results)
    r_core_arr = np.array([row['r_core'] for row in track_results]) / r_tip
    y_core_arr = np.array([row['y_core'] for row in track_results]) / r_tip
    bg = field_cache[overlay_angle]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    plot_meridional_field(bg['R'], bg['Y'], bg['v_a'], variable_label=r'mean $U_a$ [m/s]',
                           r_tip=r_tip, vmin=-18, vmax=18, ax=ax)
    ax.plot(r_core_arr, y_core_arr, '-', color='k', alpha=0.7, linewidth=1.2, zorder=2)
    sc = ax.scatter(r_core_arr, y_core_arr, c=wake_age, cmap='viridis', s=55, zorder=3,
                    edgecolors='k', linewidths=0.5)
    cbar2 = fig.colorbar(sc, ax=ax, pad=0.12)
    cbar2.set_label(r'wake age $\psi_w$ [deg]')
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_core_pressure_vs_wakeage(track_results, pref, savepath):
    '''Core-plane pressure-minimum deficit vs wake age - an independent diagnostic from
    v_swirl/Gamma/r_c entirely (find_vortex_core()'s own pressure-minimum criterion, not
    reused/refit here), so agreement with the velocity-based trend is a cross-check.'''
    wake_age = wake_age_series(track_results)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(wake_age, [row['p_min'] - pref for row in track_results], 'o-', color='k')
    ax.set_xlabel(r'wake age $\psi_w$ [deg]')
    ax.set_ylabel(r'$p_{\rm core} - p_{\infty}$ [Pa]')
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_core_pressure_coeff_vs_wakeage(track_results, pref, q_tip, savepath):
    '''Same as plot_core_pressure_vs_wakeage() but nondimensionalized by the tip dynamic pressure
    (Cp-like coefficient) - for comparing the pressure-deficit decay across different rotor
    configurations/RPMs later, same motivation as the normalized peak-swirl plot below.'''
    wake_age = wake_age_series(track_results)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(wake_age, [(row['p_min'] - pref) / q_tip for row in track_results], 'o-', color='k')
    ax.set_xlabel(r'wake age $\psi_w$ [deg]')
    ax.set_ylabel(r'$(p_{\rm core} - p_{\infty})/q_{\rm tip}$')
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_core_spread_vs_wakeage(track_results, savepath):
    '''Confidence diagnostic for the tracking itself: `spread` is the max pairwise distance
    between the 3 independent core-finding criteria (lambda2-min, p-min, |omega_theta|-max) at
    each cut (see find_vortex_core()'s own docstring) - large spread at a given wake age means
    those criteria disagree there, so treat that cut's core location (and anything derived from
    it) with less confidence, not as a hard error.'''
    wake_age = wake_age_series(track_results)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(wake_age, [row['spread'] * 1000 for row in track_results], 'o-', color='k')
    ax.set_xlabel(r'wake age $\psi_w$ [deg]')
    ax.set_ylabel(r'core-location spread [mm]')
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_peak_swirl_vs_wakeage(track_results, rpm, u_tip, savepath):
    '''
    Peak swirl velocity vs wake age, raw [m/s] + U_tip-normalized (twin y-axis, locked scale) -
    the normalized form is what makes this directly comparable across rotor configurations/RPMs.
    A viscous-diffusion reference curve (viscous_peak_swirl_decay(), fit via
    fit_viscous_peak_swirl_decay() - see their own docstrings for the derivation and explicit
    caveats) is fit directly to the measured v_swirl_max(wake age) data itself and overlaid, with
    wake age converted to a physical time via this rotor's own RPM (one revolution = 360deg =
    60/rpm seconds).
    '''
    wake_age = wake_age_series(track_results)
    v_peak_arr = np.array([row['v_swirl_max'] for row in track_results])
    t_sweep = wake_age / (6.0 * rpm)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(wake_age, v_peak_arr, 'o', color='k', label='data', markersize=6)
    try:
        amp_fit, t0_fit, r2_fit = fit_viscous_peak_swirl_decay(t_sweep, v_peak_arr)
        t_fine = np.linspace(t_sweep.min(), t_sweep.max(), 200)
        wake_age_fine = t_fine * 6.0 * rpm
        ax.plot(wake_age_fine, viscous_peak_swirl_decay(t_fine, amp_fit, t0_fit), 'r--',
                label=rf'viscous diffusion fit ($R^2={r2_fit:.2f}$)')
        print(f"viscous diffusion fit: amplitude={amp_fit:.4g}, t0={t0_fit:.4g}s, R2={r2_fit:.3f}")
    except RuntimeError as exc:
        print(f"viscous diffusion fit did not converge: {exc}")
    ax.set_xlabel(r'wake age $\psi_w$ [deg]')
    ax.set_ylabel(r'$v_{\rm swirl,max}$ [m/s]')
    ax.legend()
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$v_{\rm swirl,max}/U_{\rm tip}$')
    ax2.set_ylim(np.array(ax.get_ylim()) / u_tip)
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def vortex_profiles_at_angle(pl_master_path, field_cache, angle, rho_values):
    '''
    v_swirl/v_along-core (from field_cache's own cached velocity grid) + azimuthally-averaged
    static pressure (freshly gridded here - field_cache only stores velocity) around the tracked
    core at `angle` (must already be a field_cache key, i.e. an angle track_vortex_core_sweep()
    actually tracked).

    Returns
    -------
    v_swirl_mean, v_swirl_std, v_along_mean, v_along_std, p_mean, p_std : np.ndarray,
        shape (len(rho_values),) each
    '''
    c = field_cache[angle]
    core = c['core']
    v_swirl_mean, v_swirl_std, v_along_mean, v_along_std = azimuthal_profiles(
        c['R'], c['Y'], c['v_r'], c['v_a'], c['v_t'], core['r_core'], core['y_core'], rho_values,
    )
    p_R, p_Y, p_grid = griddata_scattered_field(
        f'{pl_master_path}/data/pstat/p_stat_{int(round(angle)):02d}deg_avg.txt', float(angle),
        n_r=250, n_y=125,
    )
    p_mean, p_std = azimuthal_scalar_profile(p_R, p_Y, p_grid, core['r_core'], core['y_core'], rho_values)
    return v_swirl_mean, v_swirl_std, v_along_mean, v_along_std, p_mean, p_std


def plot_fig5_profiles(rho_values, v_swirl_mean, v_swirl_std, v_along_mean, v_along_std,
                        savepath_tangential, savepath_axial):
    '''Quaglia et al.'s Fig. 5 pair: dimensional tangential (swirl) and axial (along-core)
    velocity profiles at a single representative wake age, with azimuthal std band - see
    vortex_profiles_at_angle() for the inputs.'''
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(rho_values * 1000, v_swirl_mean, '-', color='k')
    ax.fill_between(rho_values * 1000, v_swirl_mean - v_swirl_std, v_swirl_mean + v_swirl_std,
                     alpha=0.2, color='k')
    ax.set_xlabel(r'$\rho$ [mm]')
    ax.set_ylabel(r'$v_{\rm swirl}$ [m/s]')
    fig.tight_layout()
    fig.savefig(savepath_tangential, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(rho_values * 1000, v_along_mean, '-', color='k')
    ax.fill_between(rho_values * 1000, v_along_mean - v_along_std, v_along_mean + v_along_std,
                     alpha=0.2, color='k')
    ax.set_xlabel(r'$\rho$ [mm]')
    ax.set_ylabel(r'$v_{\rm along-core}$ [m/s]')
    fig.tight_layout()
    fig.savefig(savepath_axial, dpi=300)
    plt.close(fig)


def plot_fig6_profiles(rho_values, r_tip, u_tip, pref, profiles_by_age, age_labels, age_markers,
                        savepath_tangential, savepath_axial_deficit, savepath_pressure):
    '''
    Quaglia et al.'s Fig. 6 triplet: tangential, axial-deficit, and pressure profiles,
    dimensionless, overlaid across several wake ages.

    Parameters
    ----------
    profiles_by_age : dict keyed by an arbitrary label (e.g. 'early'/'mid'/'late') ->
        vortex_profiles_at_angle() output tuple.
    age_labels, age_markers : dict, same keys as profiles_by_age - legend label and marker style
        for each.
    '''
    rho_nd = rho_values / r_tip

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for key in profiles_by_age:
        ax.plot(rho_nd, profiles_by_age[key][0] / u_tip, age_markers[key], label=age_labels[key], markersize=5)
    ax.set_xlabel(r'$\rho/R$')
    ax.set_ylabel(r'$v_{\rm swirl}/U_{\rm tip}$')
    ax.legend(title=r'$\psi_w$')
    fig.tight_layout()
    fig.savefig(savepath_tangential, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for key in profiles_by_age:
        va = profiles_by_age[key][2]
        far_field = va[~np.isnan(va)][-1] if np.any(~np.isnan(va)) else 0.0
        ax.plot(rho_nd, (va - far_field) / u_tip, age_markers[key], label=age_labels[key], markersize=5)
    ax.set_xlabel(r'$\rho/R$')
    ax.set_ylabel(r'$(v_{\rm along-core}-v_\infty)/U_{\rm tip}$')
    ax.legend(title=r'$\psi_w$')
    fig.tight_layout()
    fig.savefig(savepath_axial_deficit, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for key in profiles_by_age:
        ax.plot(rho_nd, profiles_by_age[key][4] / pref, age_markers[key], label=age_labels[key], markersize=5)
    ax.set_xlabel(r'$\rho/R$')
    ax.set_ylabel(r'$p/p_{\rm ref}$')
    ax.legend(title=r'$\psi_w$')
    fig.tight_layout()
    fig.savefig(savepath_pressure, dpi=300)
    plt.close(fig)


def plot_fig7_tangential_models(rho_values, r_tip, u_tip, v_swirl_mean, row, models, model_labels,
                                 model_styles, model_names, savepath):
    '''Quaglia et al.'s Fig. 7a: measured swirl profile + a chosen subset of fitted vortex models
    (`model_names`), dimensionless, at a single representative (typically "late"/most-developed)
    wake age. `row` must already carry strength_<name>/rc_<name> for every name in `model_names`
    (a track_results row - see fit_vortex_model()).'''
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(rho_values / r_tip, v_swirl_mean / u_tip, '+', color='k', label='data', markersize=7)
    rho_fine = np.linspace(rho_values.min(), rho_values.max(), 200)
    for model_name in model_names:
        ax.plot(rho_fine / r_tip,
                models[model_name](rho_fine, row[f'strength_{model_name}'], row[f'rc_{model_name}']) / u_tip,
                model_styles[model_name], label=model_labels[model_name])
    ax.set_xlabel(r'$\rho/R$')
    ax.set_ylabel(r'$v_{\rm swirl}/U_{\rm tip}$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_fig8_coreradius_and_maxswirl_nd(track_results, r_tip, u_tip, savepath_coreradius,
                                          savepath_maxswirl, radius_model='lo'):
    '''Quaglia et al.'s Fig. 8 pair: core radius (from a single chosen model fit, default
    Lamb-Oseen) and peak swirl velocity vs wake age, both dimensionless.'''
    wake_age = wake_age_series(track_results)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(wake_age, [row[f'rc_{radius_model}'] / r_tip for row in track_results], 'o-', color='k')
    ax.set_xlabel(r'wake age $\psi_w$ [deg]')
    ax.set_ylabel(r'$r_c/R$')
    fig.tight_layout()
    fig.savefig(savepath_coreradius, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(wake_age, [row['v_swirl_max'] / u_tip for row in track_results], 'o-', color='k')
    ax.set_xlabel(r'wake age $\psi_w$ [deg]')
    ax.set_ylabel(r'$v_{\rm swirl,max}/U_{\rm tip}$')
    fig.tight_layout()
    fig.savefig(savepath_maxswirl, dpi=300)
    plt.close(fig)
