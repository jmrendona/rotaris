import numpy as np
from scipy.interpolate import griddata


def grid_scatter(x: np.ndarray, y: np.ndarray, values: np.ndarray, grid_resolution: int = 300):

    '''
    Grid a scattered (x, y, values) surfel field onto a regular mesh via
    linear interpolation, for pcolormesh - shared by
    FrictionLines.friction_lines()/SurfaceVariable.plot_variable_surface()'s
    own interpolate=True path (see either docstring for when this is
    worth turning on vs. leaving off - short version: a small selection,
    tens of thousands of points, not the tens of millions a full rotor
    blade's raw .snc mesh has).

    linear interpolation (not 'nearest'/'cubic'): 'nearest' doesn't
    actually smooth anything (every grid cell just copies its nearest
    raw point, the exact "blocky" look this exists to fix); 'cubic' can
    overshoot/ring near sharp real features (e.g. a leading-edge peak)
    and is noticeably slower for no benefit here. NaN (not extrapolated)
    outside the convex hull of the real (x, y) points - griddata's own
    default fill_value for 'linear' - so this never invents a value past
    the actual part's edge; pcolormesh renders NaN cells as transparent
    automatically, no separate masking needed.

    grid_resolution applies to the LONGER of the two axes' raw extent;
    the shorter axis gets proportionally fewer points, so grid cells
    stay roughly square instead of badly distorted (e.g. a vane's
    chord is often much shorter than its span - an equal point COUNT on
    both axes would make each grid cell very elongated).

    Parameters
    ----------
    x, y, values : np.ndarray, shape (n,) each
        Scattered coordinates and the field value at each one.
    grid_resolution : int
        Grid points along the longer axis - see above.

    Returns
    -------
    X, Y, Z : np.ndarray, shape (ny, nx) each
        Meshgrid coordinates and interpolated values (NaN outside the
        convex hull), ready for ax.pcolormesh(X, Y, Z, ...).
    '''

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    longer = max(x_range, y_range)

    nx = grid_resolution if x_range >= y_range else max(int(round(grid_resolution * x_range / longer)), 2)
    ny = grid_resolution if y_range >= x_range else max(int(round(grid_resolution * y_range / longer)), 2)

    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), values, (X, Y), method='linear')

    return X, Y, Z
