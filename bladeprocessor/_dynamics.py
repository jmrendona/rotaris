import numpy as np


def dynamic_pressure(rho_ref, r, rpm=None, v_ref=None):

    '''
    Reference dynamic pressure q_ref [Pa] for Cf/Cp normalization -
    shared by FrictionLines._q_ref()/SurfaceVariable._q_ref() and every
    other call site that used to inline the same rpm-based formula
    (e.g. timetrace()/periodogram()'s normalize=True paths).

    rpm given (a rotating object - a rotor blade): LOCAL q_ref =
    0.5*rho_ref*(omega*r)^2, omega = rpm*2*pi/60, r being each point's
    own radius from the rotation axis - see FrictionLines/
    SurfaceVariable's "Cf normalization" docstring for why this is
    LOCAL (varies with r) rather than one fixed velocity for the whole
    blade.

    v_ref given instead (a stationary object - a stator vane or a
    beam/strut, which has no rotation rate of its own to build a local
    dynamic pressure from): UNIFORM q_ref = 0.5*rho_ref*v_ref^2, the
    SAME value at every point regardless of r - v_ref is the known
    freestream/inlet reference velocity for this case, not derived
    from geometry at all. Broadcasts to r's own shape (scalar or
    array), so callers that pass a single point's radius (a Python
    float) get a plain float back, matching the rpm branch's own
    shape behavior.

    At least one of rpm/v_ref must be given - raises otherwise (no way
    to compute q_ref at all). v_ref WINS whenever it's set, even if rpm
    is ALSO set - that's a deliberate, legitimate combination, not an
    ambiguity: rpm can stay set purely as the phase-lock/convergence-
    checking period clock (see e.g.
    FrictionLines.plot_cf_phase_portrait's color_by='revolution') on a
    stationary object analyzed with v_ref here - the rotor's own
    blade-passing rate still drives a stationary part's unsteady loads,
    independent of which formula q_ref itself uses. Confirmed necessary
    in practice (a real stator case sets both, and erroring on that
    combination was a bug, not a safety check - fixed here).
    '''

    if rpm is None and v_ref is None:
        raise ValueError(
            "rho_ref and (rpm or v_ref) must be set (in __init__) to compute q_ref - rpm for "
            "a rotating object (q_ref = 0.5*rho_ref*(omega*r)^2, local to each point's own "
            "radius), v_ref for a stationary one (q_ref = 0.5*rho_ref*v_ref^2, uniform)."
        )

    if v_ref is not None:
        return 0.5 * rho_ref * v_ref ** 2 * np.ones_like(np.asarray(r, dtype=float))

    omega = rpm * 2 * np.pi / 60
    return 0.5 * rho_ref * (omega * r) ** 2
