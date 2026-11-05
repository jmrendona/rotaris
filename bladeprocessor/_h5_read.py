import numpy as np


def read_masked_columns(dset, row_slice, mask):

    '''
    dset[row_slice, mask], but via a single cheap CONTIGUOUS HDF5 slice
    read (bounded to the True values' own index range) followed by an
    in-memory numpy boolean filter - NOT a direct HDF5-level boolean
    fancy-index read.

    Why: HDF5 has to issue one I/O operation per CONTIGUOUS RUN of True
    values in a boolean column selection. A simple one-sided threshold
    on a coordinate that happens to correlate with the mesh's own point
    storage order (e.g. this project's span_min crop, for cases where
    it does) produces very few runs, so this was never a problem before
    azimuth_mask() existed. A geometrically-defined mask like
    azimuth_mask() has no such correlation and can have thousands of
    short runs scattered across the array - confirmed on a real case
    (a (273,) axis_direction, 85M-point DNS pressure file): a single-
    frame boolean-indexed read never completed in 15 minutes (the whole
    budget of a diagnostic job built to time it), vs. 0.35s for a FULL-
    width read of MORE data - each of those thousands of runs
    apparently paying this filesystem's own full per-request latency
    independently, not a throughput problem. This is very likely what a
    real multi-hour-"stuck" POD run was actually blocked on.

    The fix: bound the H5-level read to ONE contiguous slice spanning
    every True index (still possibly most of the array, if the mask
    isn't spatially localized - no worse than before in that case), and
    do the actual boolean filtering in-memory with numpy afterward
    (fast regardless of how scattered the mask is, since it's no longer
    talking to the filesystem per run).

    Parameters
    ----------
    dset : h5py.Dataset
    row_slice : slice or int
        Applied to dset's FIRST axis as-is (e.g. a frame slice, or a
        single frame index) - never the thing being bounded here.
    mask : np.ndarray of bool, shape (dset.shape[-1],), or slice(None)
        slice(None) (no crop) is passed through directly, matching the
        plain `dset[row_slice, :]` read it replaces exactly - every
        caller already uses slice(None) as the "no crop set" sentinel
        (see e.g. SurfaceVariable._load()'s self._surfel_mask).

    Returns
    -------
    np.ndarray - same shape dset[row_slice, mask] would have given.
    '''

    if isinstance(mask, slice):
        return dset[row_slice, mask]

    idx = np.flatnonzero(mask)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    chunk = dset[row_slice, lo:hi]

    return chunk[..., mask[lo:hi]]
