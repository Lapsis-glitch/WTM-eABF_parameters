import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# 1. Single-PMF reader (new format)
# ============================================================

def read_sequential_pmf(filename):
    """
    Reads a single PMF written by write_sequential_pmf.
    Supports 1D and ND.
    Format:
        # N
        # start step size 1
        ...
        <coords> <value>
        ...
        [blank lines allowed for ND]
    Returns:
        coords_tuple, pmf_array
    """
    with open(filename, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    # --- Parse header ---
    assert lines[0].startswith("#")
    ndim = int(lines[0].split()[1])

    starts = []
    steps = []
    sizes = []

    header_idx = 1
    for _ in range(ndim):
        parts = lines[header_idx].split()
        starts.append(float(parts[1]))
        steps.append(float(parts[2]))
        sizes.append(int(parts[3]))
        header_idx += 1

    # Skip blank line
    header_idx += 1

    # --- Build coordinate arrays ---
    coords = []
    for start, step, size in zip(starts, steps, sizes):
        coords.append(start + step * np.arange(size))
    coords_tuple = tuple(coords)

    # --- Read data ---
    data = []
    for line in lines[header_idx:]:
        if not line:
            continue
        parts = line.split()
        *coord_vals, val = parts
        data.append([float(v) for v in coord_vals] + [float(val)])

    data = np.array(data)
    pmf = data[:, -1].reshape(sizes)

    return coords_tuple, pmf


# ============================================================
# 2. Multi-PMF reader (old ABF history format)
# ============================================================

def read_sequential_pmf_blocks(filename):
    """
    Reads a PMF history file containing multiple PMFs separated by '#'.
    Each block is in the *old* format:
        <x1> <x2> ... <F>
        ...
        # new block
    Returns:
        list of (coords_tuple, pmf_array)
    """
    pmfs = []
    block = []

    def finalize_block(block):
        arr = np.array(block, float)
        coords = [np.unique(arr[:, i]) for i in range(arr.shape[1] - 1)]
        shape = tuple(len(c) for c in coords)
        pmf = arr[:, -1].reshape(shape)
        return tuple(coords), pmf

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                if block:
                    pmfs.append(finalize_block(block))
                    block = []
                continue
            if line.strip():
                block.append(line.split())

    if block:
        pmfs.append(finalize_block(block))

    return pmfs


# ============================================================
# 3. Counts reader (same multi-block format)
# ============================================================

def read_sequential_counts(filename):
    """
    Reads sampling counts in the same multi-block format as old PMFs.
    Returns:
        list_of_count_arrays, coords_tuple
    """
    blocks = read_sequential_pmf_blocks(filename)
    coords = blocks[0][0]
    counts = [pmf for (_, pmf) in blocks]
    return counts, coords


# ============================================================
# 4. Interpolation
# ============================================================

def interpolate_pmf(src_coords, src_pmf, target_coords):
    """
    Interpolate PMF from src_coords → target_coords.
    Handles ND.

    Returns:
        new_pmf : interpolated PMF on target_coords
                  (NaNs only if target is outside src domain)
    """
    # If grids match exactly, skip interpolation
    if len(src_coords) == len(target_coords) and all(
        np.array_equal(sc, tc) for sc, tc in zip(src_coords, target_coords)
    ):
        return src_pmf

    interpolator = RegularGridInterpolator(
        src_coords, src_pmf, bounds_error=False, fill_value=np.nan
    )

    mesh = np.meshgrid(*target_coords, indexing="ij")
    points = np.stack([m.flatten() for m in mesh], axis=-1)

    new_pmf = interpolator(points).reshape([len(c) for c in target_coords])

    # IMPORTANT: no NaN → max+50 here.
    # If domains match, you get no NaNs.
    # If not, NaNs will be ignored via nanmean in RMSD.
    return new_pmf


# ============================================================
# 5. PMF writer (corrected)
# ============================================================

def write_sequential_pmf(coords_tuple, pmf, filename):
    """
    Writes PMF in the new sequential format.
    - 1D: no blank lines
    - ND: blank line after each sweep of dimension 0
    """
    ndim = len(coords_tuple)
    shape = pmf.shape

    starts = [c[0] for c in coords_tuple]
    steps = [(c[1] - c[0]) if len(c) > 1 else 0.0 for c in coords_tuple]
    sizes = [len(c) for c in coords_tuple]

    with open(filename, "w") as f:
        f.write(f"# {ndim}\n")
        for start, step, size in zip(starts, steps, sizes):
            f.write(f"# {start: .14e}  {step: .14e}  {size:8d}  1\n")
        f.write("\n")

        if ndim == 1:
            x = coords_tuple[0]
            for i in range(shape[0]):
                f.write(f"{x[i]: .14e}   {pmf[i]: .14e}\n")
        else:
            for i in range(shape[0]):
                it = np.nditer(pmf[i], flags=['multi_index'])
                for val in it:
                    idx = (i,) + it.multi_index
                    coords = [coords_tuple[d][idx[d]] for d in range(ndim)]
                    f.write("  ".join(f"{c: .14e}" for c in coords) + f"   {val: .14e}\n")
                f.write("\n")