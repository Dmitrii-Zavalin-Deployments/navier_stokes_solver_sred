# src/step5/io_archivist.py

from pathlib import Path
import h5py
import numpy as np
from src.common.field_schema import FI

def save_snapshot(state):
    """
    Exports the physical 3D domain state to HDF5.
    
    Compliance:
    - Direct Foundation Slicing: Uses the FI schema to extract data directly 
      from the pre-allocated fields_buffer, bypassing object-pointer overhead.
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"snapshot_{state.iteration:04d}.h5"
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    # Access the contiguous Foundation buffer directly
    # Shape is (N_CELLS, FI.num_fields())
    data = state.fields_buffer 

    with h5py.File(filename, 'w') as h5f:
        # Physical Fields extracted via direct schema-locked slicing
        # This is an O(1) memory operation relative to the object graph
        h5f.create_dataset('vx', data=data[:, FI.VX].reshape(nx, ny, nz))
        h5f.create_dataset('vy', data=data[:, FI.VY].reshape(nx, ny, nz))
        h5f.create_dataset('vz', data=data[:, FI.VZ].reshape(nx, ny, nz))
        h5f.create_dataset('p',  data=data[:, FI.P].reshape(nx, ny, nz))
        
        # Spatial/Mask metadata (Assuming these are cached in state.grid or equivalent)
        h5f.create_dataset('x', data=state.grid.x_mesh)
        h5f.create_dataset('y', data=state.grid.y_mesh)
        h5f.create_dataset('z', data=state.grid.z_mesh)
        h5f.create_dataset('mask', data=state.grid.mask_mesh)
        
        # Global Metadata
        h5f.attrs['time'] = state.time
        h5f.attrs['iteration'] = state.iteration
        h5f.attrs['dx'] = state.grid.dx
        h5f.attrs['dy'] = state.grid.dy
        h5f.attrs['dz'] = state.grid.dz
    
    state.manifest.saved_snapshots.append(str(filename))