# src/common/grid_math.py

def get_flat_index(i: int, j: int, k: int, nx_buf: int, ny_buf: int) -> int:
    # Stride for k is implicitly (nx_buf * ny_buf)
    return (i + 1) + nx_buf * (j + 1) + (nx_buf * ny_buf) * (k + 1)

def get_coords_from_index(index: int, nx_buf: int, ny_buf: int) -> tuple[int, int, int]:
    # Calculate the area of the XY plane
    xy_plane = nx_buf * ny_buf
    
    k = (index // xy_plane) - 1
    rem = index % xy_plane
    j = (rem // nx_buf) - 1
    i = (rem % nx_buf) - 1
    
    return i, j, k
