"""
Post-fix sanity check for the Surface_X/Y/Z-Force reference-frame fix
(converters/snc_reader.py) - see HANDOFF.md's OPEN INVESTIGATION.

An EARLIER version of this script applied the rotation as a POST-HOC
correction to an already-converted, still-buggy forces_out.h5, since
that file's source .snc no longer existed on HPC scratch. Both
forces_out.h5 AND its source .snc are now gone entirely, so that
approach is no longer usable - and it's no longer needed anyway, since
the fix now lives inside SNCReader.to_h5() itself: Surface_X/Y/Z-Force
in ANY freshly-converted file is already rotated into the LRF. This
version checks a fresh conversion directly instead of correcting an old
one.

What it checks: the net in-plane wall-shear angle (atan2 of the summed
spanwise/chordwise wall-shear components over a span-cropped patch),
tracked across a frame range spanning a meaningful chunk of a
revolution. BEFORE the fix, this swept a full ~317-330 deg once per
revolution (HANDOFF's evidence #1, measured against the now-deleted
forces_out.h5) - a vector genuinely expressed in the blade's OWN (LRF)
frame should NOT do this; it should stay small/flat, modulo real
unsteady flow effects (e.g. the genuine once-per-revolution near-wall
shear periodicity already confirmed separately - see HANDOFF's
"RESOLVED: Cf-magnitude-growth investigation"). If a freshly-converted
file STILL shows a large, nearly-360-deg-per-revolution sweep, the fix
isn't taking effect (wrong sign in _write_surfel_group(), or this file
wasn't actually reconverted with the fixed code) - if it stays small,
the fix is confirmed on real data.

If Metadata/lrf_position_rad is present (i.e. this file was converted
with nc_stats_path), the script also reports how much the LRF itself
actually rotated over the same frame range, for a direct, concrete
comparison: "the blade turned this many degrees; the measured wall-shear
direction only wobbled by this many" is the qualitative pass criterion.

Usage:
    python verify_rotation_fix.py /path/to/freshly_converted_forces.h5 \\
        --span-min 0.02 --frame-start 0 --frame-end 176 --frame-step 4
"""
import argparse
import numpy as np
import h5py


def net_angle(tau_chord, tau_span, mask):
    c = tau_chord[:, mask].sum(axis=1)
    s = tau_span[:, mask].sum(axis=1)
    return np.degrees(np.unwrap(np.arctan2(s, c)))


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('h5_path')
    ap.add_argument('--span-min', type=float, default=0.02)
    ap.add_argument('--span-axis', type=int, default=0)
    ap.add_argument('--chord-axis', type=int, default=2)
    ap.add_argument('--frame-start', type=int, default=0)
    ap.add_argument('--frame-end', type=int, default=176)
    ap.add_argument('--frame-step', type=int, default=4)
    args = ap.parse_args()

    with h5py.File(args.h5_path, 'r') as f:

        n_frames = f['Metadata/frame_index'].shape[0]
        has_lrf_position = 'Metadata/lrf_position_rad' in f
        lrf_position_rad = f['Metadata/lrf_position_rad'][:] if has_lrf_position else None

        labels = ['Upper', 'Lower'] if 'Upper' in f['Geometry'] else [None]

        for label in labels:

            geo_path = f'Geometry/{label}' if label else 'Geometry'
            data_path = f'Data/{label}' if label else 'Data'
            geo, data = f[geo_path], f[data_path]

            positions = np.column_stack([geo['X'][:], geo['Y'][:], geo['Z'][:]])
            normals = np.column_stack([geo['Normal_X'][:], geo['Normal_Y'][:], geo['Normal_Z'][:]])
            force = np.stack([
                data['Surface_X-Force'][:], data['Surface_Y-Force'][:], data['Surface_Z-Force'][:],
            ], axis=-1)  # (n_frames, n_points, 3) - already rotated into the LRF by to_h5()

            span = positions[:, args.span_axis]
            span = span - (span.min() + span.max()) / 2
            mask = span >= args.span_min

            frames = list(range(args.frame_start, min(args.frame_end, n_frames), args.frame_step))
            if len(frames) < 2:
                raise ValueError(
                    f"Only {len(frames)} frame(s) selected ({args.frame_start}..{args.frame_end} "
                    f"step {args.frame_step}, n_frames={n_frames}) - need at least 2 to measure a sweep."
                )

            f_normal = np.einsum('fpc,pc->fp', force[frames], normals)
            tau = force[frames] - f_normal[..., None] * normals[None, :, :]
            angle = net_angle(tau[..., args.chord_axis], tau[..., args.span_axis], mask)

            print(f"\n=== {label or 'unsplit'} ===")
            print(f"Wall-shear angle: {angle[0]:.1f} -> {angle[-1]:.1f} deg "
                  f"(sweep {angle.max() - angle.min():.1f} deg over {len(frames)} sampled frames, "
                  f"{frames[0]}..{frames[-1]})")

            if has_lrf_position:
                lrf_deg = np.degrees(lrf_position_rad[frames])
                lrf_sweep = abs(lrf_deg[-1] - lrf_deg[0])
                print(f"Actual LRF rotation over the same range: {lrf_sweep:.1f} deg")
                print("  -> if the fix is correct, the wall-shear sweep above should be MUCH "
                      "smaller than this (residual unsteady wobble, not a rotation artifact); "
                      "if the two numbers are close, the fix isn't taking effect on this file.")
            else:
                print("(Metadata/lrf_position_rad not present - this file wasn't converted with "
                      "nc_stats_path, so there's no independent 'how much did the blade actually "
                      "rotate' figure to compare against here. Still meaningful on its own: "
                      "HANDOFF's evidence #1 measured a ~317-330 deg sweep BEFORE the fix, over a "
                      "similar frame range/step on a similar case - a sweep anywhere near that "
                      "range here means the fix isn't taking effect; a small sweep (a few tens of "
                      "degrees at most) means it is.)")


if __name__ == '__main__':
    main()
