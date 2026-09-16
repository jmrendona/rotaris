import argparse
import glob
import os
import re
import h5py
import numpy as np


def _parse_chunk_range(path: str):

    '''
    Extract (first_frame, last_frame) from a
    <prefix>_frames_<first>_<last>.h5 filename (see
    submit_pressure_chunks.sh's own naming convention) - this is the SORT
    KEY that determines merge order, not just a cosmetic label, so a
    filename that doesn't match this pattern is refused rather than
    silently merged in whatever order glob() happened to return it.
    '''

    m = re.search(r'_frames_(\d+)_(\d+)\.h5$', os.path.basename(path))
    if not m:
        raise ValueError(
            f"'{path}' doesn't match the expected <prefix>_frames_<first>_<last>.h5 naming "
            "(see submit_pressure_chunks.sh) - can't determine merge order from it."
        )
    return int(m.group(1)), int(m.group(2))


def merge_h5_chunks(chunk_paths, output_path: str):

    '''
    Merge N chunk `.h5` files - each written by convert_snc_to_h5() (or
    SNCReader.to_h5()) for a disjoint frame range, e.g. from
    submit_pressure_chunks.sh - into ONE combined `.h5`, in the SAME
    schema (Geometry/Data/Metadata) a single unchunked conversion of the
    whole range would have produced. Every downstream consumer
    (FrictionLines, SurfaceVariable, StripForces) needs zero changes to
    read the result - it's exactly the SNCReader.to_h5()/
    EnsightSeriesWriter output shape, just assembled from pieces.

    Deliberately does more validation than a plain concatenation would,
    given this project has already found more than one real case this
    session of "should be identical/contiguous" silently NOT being so:

    - Chunks are ordered by frame range PARSED FROM THE FILENAME (see
      _parse_chunk_range), not by however glob()/the filesystem happens
      to list them.
    - Frame coverage is checked CONTIGUOUS (no gap, overlap, or
      duplicate) across the sorted chunks purely from filenames, AND
      each chunk's own actual `Metadata/frame_index` content is checked
      against what its filename claims - catching a stale/mislabeled
      chunk file whose filename doesn't match what's actually inside it.
    - Geometry (frame-independent - written once per chunk file, from
      that chunk's own reference frame) is taken from the first chunk
      and verified BYTE-IDENTICAL against every other chunk's own copy
      before being trusted - if the mesh really is invariant across the
      whole range (true for this project's own rigid-blade cases), this
      always passes; if it doesn't, that's exactly the kind of thing
      worth knowing about before silently picking one chunk's geometry
      as if it were universal.

    Streams one variable's data at a time straight from each source
    chunk into the right row-range of the output file - never holds more
    than one chunk's one variable in memory at once, since the combined
    output for a real case can be hundreds of GB (holding it all in RAM
    is not an option).

    Parameters
    ----------
    chunk_paths : list of str
        Paths to the chunk `.h5` files to merge (any order - re-sorted
        internally by parsed frame range).
    output_path : str
        Path to the combined `.h5` file to create.
    '''

    if len(chunk_paths) < 2:
        raise ValueError(f"Need at least 2 chunk files to merge, got {len(chunk_paths)}.")

    ranges = {p: _parse_chunk_range(p) for p in chunk_paths}
    ordered = sorted(chunk_paths, key=lambda p: ranges[p])

    # --- frame coverage: contiguous, no gaps/overlaps/duplicates, purely from filenames ---
    expected_start = ranges[ordered[0]][0]
    for path in ordered:
        start, end = ranges[path]
        if start != expected_start:
            raise ValueError(
                f"Frame coverage gap or overlap in chunk filenames: expected the next chunk to "
                f"start at frame {expected_start}, but '{os.path.basename(path)}' starts at {start}. "
                "Check for a missing or duplicate chunk file before merging."
            )
        if end < start:
            raise ValueError(f"'{os.path.basename(path)}' has last frame {end} < first frame {start}.")
        expected_start = end + 1

    total_frames = ranges[ordered[-1]][1] - ranges[ordered[0]][0] + 1
    global_first = ranges[ordered[0]][0]

    print(f"Merging {len(ordered)} chunks covering frames [{global_first},{ranges[ordered[-1]][1]}] "
          f"({total_frames} frames total) into '{output_path}'")

    # --- each chunk's ACTUAL frame_index content must match what its filename claims ---
    for path in ordered:
        start, end = ranges[path]
        with h5py.File(path, 'r') as f:
            actual = f['Metadata/frame_index'][:]
        expected = np.arange(start, end + 1)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                f"'{os.path.basename(path)}''s filename claims frames [{start},{end}], but its own "
                f"Metadata/frame_index is {actual.tolist()} - this chunk's filename doesn't match "
                "what's actually inside it. Refusing to merge until this is understood (stale file "
                "from a re-run with different --first/--last, most likely)."
            )

    with h5py.File(ordered[0], 'r') as f0:
        labels = list(f0['Geometry'].keys())
        variable_names = {label: list(f0[f'Data/{label}'].keys()) for label in labels}
        n_points = {label: f0[f'Geometry/{label}/X'].shape[0] for label in labels}
        dtype_by_var = {
            label: {v: f0[f'Data/{label}/{v}'].dtype for v in variable_names[label]}
            for label in labels
        }
        axis_origin = f0['Metadata/lrf_axis_origin'][:]
        axis_direction = f0['Metadata/lrf_axis_direction'][:]
        meta_keys = [k for k in f0['Metadata'].keys() if k not in ('lrf_axis_origin', 'lrf_axis_direction')]
        meta_dtypes = {k: f0[f'Metadata/{k}'].dtype for k in meta_keys}
        geometry_by_label = {
            label: {coord: f0[f'Geometry/{label}/{coord}'][:] for coord in ('X', 'Y', 'Z')}
            for label in labels
        }

    # --- verify every OTHER chunk's geometry matches the first one, byte-for-byte ---
    for path in ordered[1:]:
        with h5py.File(path, 'r') as f:
            for label in labels:
                for coord in ('X', 'Y', 'Z'):
                    if not np.array_equal(f[f'Geometry/{label}/{coord}'][:], geometry_by_label[label][coord]):
                        raise ValueError(
                            f"Geometry mismatch: '{os.path.basename(path)}''s Geometry/{label}/{coord} "
                            f"differs from '{os.path.basename(ordered[0])}''s - these chunks don't look "
                            "like they came from the same mesh/conversion; refusing to merge blindly."
                        )
    print(f"Geometry verified identical across all {len(ordered)} chunks.")

    with h5py.File(output_path, 'w') as out:

        geo_group = out.create_group('Geometry')
        for label in labels:
            geo = geo_group.create_group(label)
            for coord in ('X', 'Y', 'Z'):
                geo.create_dataset(coord, data=geometry_by_label[label][coord])

        data_group = out.create_group('Data')
        out_data_dsets = {}
        for label in labels:
            dgroup = data_group.create_group(label)
            out_data_dsets[label] = {
                var: dgroup.create_dataset(var, shape=(total_frames, n_points[label]),
                                            dtype=dtype_by_var[label][var])
                for var in variable_names[label]
            }

        meta_group = out.create_group('Metadata')
        meta_group.create_dataset('lrf_axis_origin', data=axis_origin)
        meta_group.create_dataset('lrf_axis_direction', data=axis_direction)
        out_meta_dsets = {
            k: meta_group.create_dataset(k, shape=(total_frames,), dtype=meta_dtypes[k])
            for k in meta_keys
        }

        row = 0
        for path in ordered:
            start, end = ranges[path]
            n = end - start + 1
            print(f"  copying frames [{start},{end}] from '{os.path.basename(path)}' -> rows [{row},{row + n - 1}]")
            with h5py.File(path, 'r') as f:
                for label in labels:
                    for var in variable_names[label]:
                        out_data_dsets[label][var][row:row + n, :] = f[f'Data/{label}/{var}'][:]
                for k in meta_keys:
                    out_meta_dsets[k][row:row + n] = f[f'Metadata/{k}'][:]
            row += n

    total_points = sum(n_points.values())
    print(f"Wrote '{output_path}' - {total_frames} frames, {total_points} total points across "
          f"{len(labels)} surface group(s) ({', '.join(labels)}).")


def main():

    parser = argparse.ArgumentParser(
        description='Merge chunked <prefix>_frames_<first>_<last>.h5 files (see '
                    'submit_pressure_chunks.sh) into one combined .h5, same schema as an '
                    'unchunked conversion.'
    )
    parser.add_argument('output_path', help='Path to the combined HDF5 file to create.')
    parser.add_argument('--chunks-glob', required=True,
                         help='Shell-quoted glob pattern matching the chunk files, e.g. '
                              '"/path/to/pressure_frames_*.h5" (quote it so the shell does not '
                              'expand it first - this script does its own globbing/ordering).')
    args = parser.parse_args()

    chunk_paths = sorted(glob.glob(args.chunks_glob))
    if not chunk_paths:
        raise ValueError(f"No files matched --chunks-glob '{args.chunks_glob}'.")

    merge_h5_chunks(chunk_paths, args.output_path)


if __name__ == '__main__':
    main()
