"""
Decisive sign/magnitude check for the Surface_X/Y/Z-Force reference-frame
fix (converters/snc_reader.py) - see HANDOFF.md's OPEN INVESTIGATION.

Why this exists as a STANDALONE script rather than just re-running
SNCReader.to_h5() on the raw .snc: the 829-frame forces_out.h5
(/scratch/jmrendon/Rotor-alone/6e-5_6000rpm/forces_out.h5) that HANDOFF's
angle-sweep evidence was gathered against was converted from a raw .snc
that no longer exists on HPC scratch (deleted to save space - see
HANDOFF's "File/data conventions"). Without that source file, the fixed
to_h5() can't be re-run to produce a corrected forces_out.h5 directly.

This script instead applies the IDENTICAL rotation math directly to the
EXISTING (buggy) forces_out.h5's already-stored Surface_X/Y/Z-Force
arrays - a post-hoc correction, not a re-conversion - and re-runs the
exact net-in-plane-angle-vs-frame sweep test HANDOFF describes, so the
fix's sign can be confirmed against the real 829-frame/~4.6-revolution
case without needing the original .snc.

Two ways to get the per-frame rotation angle, in order of preference:

1. If you still have (or can regenerate) an `exaritool nc-stats.ri
   <snc> -detail` dump for this exact case, pass --nc-stats and this
   uses PowerFLOW's own authoritative lrf_position_rad directly (see
   converters.snc_reader.parse_nc_stats()) - most trustworthy, no
   assumptions.
2. Otherwise, pass --omega-deg-per-frame (default -2.0, matching
   HANDOFF's evidence #1/#3 for this case) - ASSUMES forces_out.h5 was
   converted at the same per-frame dt as the small 2-frame raw .snc
   (f50_SMF_forces_rotor.snc / 2f_SMF_forces_rotor.snc) evidence #3 was
   measured from. Confirm this assumption holds (same case, same
   conversion frame stride) before trusting the result if you use this
   path - if forces_out.h5 used a different --first/--last stride, the
   per-frame angle differs and this would silently give a wrong rate.

Usage:
    python verify_rotation_fix.py /scratch/jmrendon/Rotor-alone/6e-5_6000rpm/forces_out.h5 \\
        --span-min 0.02 --chord-axis 2 --span-axis 0 \\
        [--nc-stats /path/to/nc_stats.txt] [--omega-deg-per-frame -2.0]

Expected (from HANDOFF's evidence #1/#2, if the sign in the fix is
right): BEFORE correction, the net in-plane wall-shear angle should
sweep close to the previously-measured ~317-330 deg over one revolution
(frames 0-176, step 4). AFTER correction (this script's output), that
sweep should collapse to roughly the ~90-110 deg residual HANDOFF
already measured by hand - if instead it BALLOONS (e.g. toward
~650-680 deg, per HANDOFF's own note on what the wrong sign does),
the sign in _rotation_angle()/_write_surfel_group() (currently
"rotate by -angle(frame)") needs flipping, not the rate itself.
"""
import argparse
import sys
import numpy as np
import h5py

sys.path.insert(0, '.')
from converters.snc_reader import SNCReader, parse_nc_stats  # noqa: E402


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
    ap.add_argument('--nc-stats', default=None)
    ap.add_argument('--omega-deg-per-frame', type=float, default=-2.0)
    args = ap.parse_args()

    with h5py.File(args.h5_path, 'r') as f:

        axis_direction = f['Metadata/lrf_axis_direction'][:]
        axis_direction = axis_direction / np.linalg.norm(axis_direction)
        n_frames = f['Metadata/frame_index'].shape[0]

        labels = ['Upper', 'Lower'] if 'Upper' in f['Geometry'] else [None]

        results = {}

        for label in labels:

            geo_path = f'Geometry/{label}' if label else 'Geometry'
            data_path = f'Data/{label}' if label else 'Data'
            geo, data = f[geo_path], f[data_path]

            positions = np.column_stack([geo['X'][:], geo['Y'][:], geo['Z'][:]])
            normals = np.column_stack([geo['Normal_X'][:], geo['Normal_Y'][:], geo['Normal_Z'][:]])
            force = np.stack([
                data['Surface_X-Force'][:], data['Surface_Y-Force'][:], data['Surface_Z-Force'][:],
            ], axis=-1)  # (n_frames, n_points, 3)

            span = positions[:, args.span_axis]
            span = span - (span.min() + span.max()) / 2
            mask = span >= args.span_min

            frames = list(range(args.frame_start, min(args.frame_end, n_frames), args.frame_step))

            if args.nc_stats:
                frame_meta = parse_nc_stats(args.nc_stats)
                angles_deg = np.degrees([frame_meta[fr]['lrf_position_rad'] for fr in frames])
            else:
                angles_deg = args.omega_deg_per_frame * np.array(frames, dtype=float)

            f_normal = np.einsum('fpc,pc->fp', force[frames], normals)
            tau = force[frames] - f_normal[..., None] * normals[None, :, :]
            tau_chord = tau[..., args.chord_axis]
            tau_span = tau[..., args.span_axis]

            before = net_angle(tau_chord, tau_span, mask)

            # Post-hoc correction: rotate force about axis_direction by
            # -angle(frame) BEFORE re-deriving tau - same operation
            # SNCReader.to_h5()'s fix now applies at conversion time.
            corrected_force = np.empty_like(force[frames])
            for i, ang in enumerate(np.radians(angles_deg)):
                corrected_force[i] = SNCReader._rotate_about_axis(force[frames[i]], axis_direction, -ang)

            f_normal_c = np.einsum('fpc,pc->fp', corrected_force, normals)
            tau_c = corrected_force - f_normal_c[..., None] * normals[None, :, :]
            after = net_angle(tau_c[..., args.chord_axis], tau_c[..., args.span_axis], mask)

            results[label or 'unsplit'] = (frames, before, after)

    for label, (frames, before, after) in results.items():
        print(f"\n=== {label} ===")
        print(f"BEFORE correction: {before[0]:.1f} -> {before[-1]:.1f} deg "
              f"(span {before[-1] - before[0]:.1f} deg)")
        print(f"AFTER  correction: {after[0]:.1f} -> {after[-1]:.1f} deg "
              f"(span {after[-1] - after[0]:.1f} deg)")
        print("Expect BEFORE span ~317-330 deg (matches HANDOFF's prior measurement) and "
              "AFTER span to COLLAPSE to roughly ~90-110 deg if the fix's sign is right - "
              "if AFTER instead balloons (e.g. toward ~650-680 deg), the sign needs flipping.")


if __name__ == '__main__':
    main()
