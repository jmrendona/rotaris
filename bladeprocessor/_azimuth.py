import numpy as np


def azimuth_deg(positions: np.ndarray, axis_origin: np.ndarray, axis_direction: np.ndarray) -> np.ndarray:

    '''
    Azimuth angle [deg], in (-180, 180], of each position around the
    rotor's own rotation axis - 0 deg along an arbitrary but FIXED
    in-plane reference direction (global X, or Y if axis_direction is
    too close to X), so repeated calls against the same axis_direction
    always agree on what "0 deg" means.

    Built for separating a .snc "rotor" face's surfels by which physical
    BLADE they belong to, when (as found for a multi-blade mesh with no
    per-blade face_name - see FrictionLines/SurfaceVariable/StripForces'
    azimuth_center_deg parameter) every blade shares the same face and
    the same radius range, so neither face_name nor a Cartesian
    span_min/span_max crop can tell them apart - azimuth can, since each
    blade occupies its own distinct, tight azimuth band around the
    rotation axis.

    Parameters
    ----------
    positions : np.ndarray, shape (n, 3)
    axis_origin : np.ndarray, shape (3,)
        A point on the rotation axis (e.g. Metadata/lrf_axis_origin).
    axis_direction : np.ndarray, shape (3,)
        Unit vector along the rotation axis (e.g.
        Metadata/lrf_axis_direction, normalized).

    Returns
    -------
    np.ndarray, shape (n,)
    '''

    rel = positions - axis_origin
    along = rel @ axis_direction
    radial_vec = rel - along[:, None] * axis_direction

    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, axis_direction)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = reference - np.dot(reference, axis_direction) * axis_direction
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_direction, e1)

    c1 = radial_vec @ e1
    c2 = radial_vec @ e2

    return np.degrees(np.arctan2(c2, c1))


def azimuth_mask(positions: np.ndarray, axis_origin: np.ndarray, axis_direction: np.ndarray,
                  azimuth_center_deg: float, azimuth_tol_deg: float) -> np.ndarray:

    '''
    Boolean mask, True for every position within azimuth_tol_deg of
    azimuth_center_deg (wrap-aware across the +-180 deg seam) - see
    azimuth_deg() for the angle convention.

    Returns
    -------
    np.ndarray of bool, shape (n,)
    '''

    az = azimuth_deg(positions, axis_origin, axis_direction)
    delta = (az - azimuth_center_deg + 180) % 360 - 180

    return np.abs(delta) <= azimuth_tol_deg
