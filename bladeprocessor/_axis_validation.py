import numpy as np


def validate_chord_span_thickness_axes(normals: np.ndarray, span_axis: int, chord_axis: int,
                                        thickness_axis: int, high: float = 0.7, low: float = 0.3):

    '''
    Sanity-check a span_axis/chord_axis/thickness_axis assignment against
    the file's own per-surfel normals, before trusting it for anything.

    Physical basis: a lifting blade's local surface normal is always
    close to its own thickness direction, by definition (that's what
    "thickness direction" means for a thin, nearly-flat body) - so
    thickness_axis should have a HIGH mean |normal component|, while
    span_axis/chord_axis (both roughly IN the blade surface) should both
    be LOW. This holds regardless of blade pitch, twist, or sweep - it's
    a geometric fact about thin bodies, not a convention - so it catches
    a mesh whose axis convention doesn't match this project's default
    (span_axis=0, chord_axis=2, thickness_axis=1, validated only against
    the original isolated-rotor case) BEFORE it produces a silently
    wrong (e.g. squashed-looking) plot, rather than after.

    Deliberately does NOT try to auto-detect/correct chord vs span on
    its own, even though it could suggest a "best" axis - a rotation-
    axis-derived chord was tried and abandoned early on in this project
    because it broke under real blade pitch/twist (see FrictionLines'
    class docstring); only reports what the data suggests via the raised
    error, never silently applies it.

    Parameters
    ----------
    normals : np.ndarray, shape (n, 3)
        Per-surfel unit normal vectors, already loaded.
    span_axis, chord_axis, thickness_axis : int
        The configured assignment to check.
    high, low : float
        Thresholds - thickness_axis must have mean |normal component|
        >= high; span_axis/chord_axis must both be <= low. The gap
        between them (0.3-0.7) is deliberately left as "also an error" -
        an ambiguous case shouldn't silently pass either.

    Raises
    ------
    ValueError
        If the configured assignment doesn't match the geometry - the
        message reports the actual mean |normal component| per raw axis
        and which one looks like the true thickness/normal direction.
    '''

    mean_alignment = np.abs(normals).mean(axis=0)  # (3,) - mean |N_axis| per raw axis (X, Y, Z)
    axis_label = {0: 'X', 1: 'Y', 2: 'Z'}

    configured = {'thickness_axis': thickness_axis, 'span_axis': span_axis, 'chord_axis': chord_axis}
    problems = []

    if mean_alignment[thickness_axis] < high:
        problems.append(
            f"thickness_axis={thickness_axis} ({axis_label[thickness_axis]}) has mean "
            f"|normal component|={mean_alignment[thickness_axis]:.3f} - expected >= {high} (a blade's "
            "local normal should be close to its own thickness direction)."
        )
    for name in ('span_axis', 'chord_axis'):
        idx = configured[name]
        if mean_alignment[idx] > low:
            problems.append(
                f"{name}={idx} ({axis_label[idx]}) has mean |normal component|={mean_alignment[idx]:.3f} "
                f"- expected <= {low} ({name.replace('_axis', '')} should lie roughly IN the blade "
                "surface, not along its normal)."
            )

    if not problems:
        return

    best_axis = int(np.argmax(mean_alignment))
    raise ValueError(
        "span_axis/chord_axis/thickness_axis don't match this file's actual geometry: "
        + " ".join(problems)
        + f" Mean |normal component| per raw axis: X={mean_alignment[0]:.3f}, "
          f"Y={mean_alignment[1]:.3f}, Z={mean_alignment[2]:.3f} - axis {best_axis} "
          f"({axis_label[best_axis]}) looks like the true thickness/normal direction for this file. "
          "Adjust span_axis/chord_axis/thickness_axis accordingly (span vs. chord between the two "
          "remaining axes still needs to be told apart independently - e.g. by comparing each "
          "candidate's raw range against the known span/chord length - this check only identifies "
          "thickness, it doesn't guess between the other two). Pass validate_axes=False to skip this "
          "check if you're confident it doesn't apply to this file."
    )
