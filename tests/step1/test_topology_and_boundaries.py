# tests/step1/test_topology_and_boundaries.py

import pytest
import numpy as np
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from src.step1.map_geometry_mask import map_geometry_mask
from src.step1.assemble_simulation_state import assemble_simulation_state
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_grid():
    """Provides canonical grid metadata (Section 5 Compliance)."""
    return solver_input_schema_dummy()["grid"]

@pytest.fixture
def dummy_data():
    """Provides the full canonical dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

@pytest.fixture
def valid_6_face_bc():
    """
    Provides a complete, mathematically sound 6-face BC list.
    Satisfies 'The Zero-Debt Closure Rule' (Phase F, Rule 12).
    """
    return [
        {"location": "x_min", "type": "inflow", "values": {"u": 123.45, "v": 0.0, "w": 0.0}},
        {"location": "x_max", "type": "pressure", "values": {"p": 0.0}},
        {"location": "y_min", "type": "no-slip", "values": {}},
        {"location": "y_max", "type": "no-slip", "values": {}},
        {"location": "z_min", "type": "no-slip", "values": {}},
        {"location": "z_max", "type": "no-slip", "values": {}}
    ]

# --- SECTION 1: BOUNDARY CONDITION COMPLIANCE (Phase F, Rule 12) ---

def test_full_bc_normalization_and_storage(dummy_grid, valid_6_face_bc):
    """
    Verifies 6-face closure and Loud Value Traceability.
    Uses '123.45' to ensure values are not defaults.
    """
    bc_table = parse_boundary_conditions(valid_6_face_bc, dummy_grid)
    
    # 1. Zero-Debt Closure: Cuboid must have 6 faces
    assert len(bc_table) == 6
    
    # 2. Loud Value Traceability: Verify the unique prime 123.45
    assert isinstance(bc_table["x_min"]["u"], float)
    assert bc_table["x_min"]["u"] == 123.45
    assert bc_table["x_max"]["p"] == 0.0

def test_invalid_location_override(dummy_grid):
    """Verifies error for non-canonical location names (Domain Extent Guard)."""
    bc_list = [{"location": "center_of_universe", "type": "no-slip", "values": {}}]
    with pytest.raises(ValueError, match="(?i)Invalid or missing boundary location"):
        parse_boundary_conditions(bc_list, dummy_grid)

def test_inflow_requires_numeric_uvw(dummy_grid):
    """Compliance: Inflow requires a full 3D velocity vector [u, v, w]."""
    incomplete = [{"location": "x_min", "type": "inflow", "values": {"u": 5.0}}]
    with pytest.raises(ValueError, match="requires numeric velocity"):
        parse_boundary_conditions(incomplete, dummy_grid)

def test_pressure_requires_numeric_p(dummy_grid):
    """Compliance: Pressure type must define scalar 'p'."""
    missing_p = [{"location": "x_max", "type": "pressure", "values": {"u": 0.0}}]
    with pytest.raises(ValueError, match="requires numeric 'p'"):
        parse_boundary_conditions(missing_p, dummy_grid)

def test_physical_exclusion_collision(dummy_grid):
    """
    Phase F Mandate: Collision Prevention.
    Outflow/Slip/No-Slip cannot define pressure (Over-specification).
    """
    # Outflow defining pressure is a constitutional violation
    bad_outflow = [{"location": "x_max", "type": "outflow", "values": {"p": 10.0}}]
    with pytest.raises(ValueError, match="cannot define pressure"):
        parse_boundary_conditions(bad_outflow, dummy_grid)

def test_duplicate_location_logic(dummy_grid):
    """Prevention of physics collisions: one face cannot have two types."""
    duplicate_loc = [
        {"location": "y_min", "type": "no-slip"},
        {"location": "y_min", "type": "free-slip"}
    ]
    with pytest.raises(ValueError, match="(?i)Duplicate BC"):
        parse_boundary_conditions(duplicate_loc, dummy_grid)

def test_incomplete_domain_trigger(dummy_grid):
    """Trigger: The Zero-Debt Closure Rule (Line 64)."""
    incomplete = [{"location": "x_min", "type": "no-slip"}]
    with pytest.raises(ValueError, match="Incomplete Domain"):
        parse_boundary_conditions(incomplete, dummy_grid)

# --- SECTION 2: GEOMETRY & MASKING (Forbidden Topology) ---



def test_mask_reshaping_fortran_order(dummy_data):
    """
    Verifies Fortran-order mapping (i-fastest).
    Requirement: Maintain absolute schema symmetry.
    """
    grid = dummy_data["grid"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    total_cells = nx * ny * nz

    for target_index in [0, 1, nx, nx * ny - 1, total_cells - 1]:
        flat = [0] * total_cells
        flat[target_index] = 1
        
        mask_3d, _, _ = map_geometry_mask(flat, grid)

        # 3D unravelling logic (Fortran order / Column-major)
        k_exp = target_index // (nx * ny)
        remainder = target_index % (nx * ny)
        j_exp = remainder // nx
        i_exp = remainder % nx

        assert mask_3d[i_exp, j_exp, k_exp] == 1, f"Mapping failed at index {target_index}"

def test_forbidden_topology_rule(dummy_data):
    """
    Phase F Mandate: The 'Forbidden Topology' Rule.
    Geometry mask is restricted to the set {-1, 0, 1}.
    """
    grid = dummy_data["grid"]
    total_cells = grid["nx"] * grid["ny"] * grid["nz"]

    # Unauthorized value (5) triggers safety firewall
    bad_val_flat = [0] * total_cells
    bad_val_flat[0] = 5
    with pytest.raises(ValueError, match="Mask contains unauthorized values"):
        map_geometry_mask(bad_val_flat, grid)

def test_mask_validation_mismatch(dummy_data):
    """Trigger: Mask length must exactly match product of dimensions."""
    grid = dummy_data["grid"]
    with pytest.raises(ValueError, match="Mask length mismatch"):
        map_geometry_mask([1, 0, 1], grid)

# --- SECTION 3: INTEGRITY & ASSEMBLY ---

def test_assemble_state_spatial_incoherence():
    """Trigger: Spatial Incoherence (Mismatched mask dimensions)."""
    grid = {"nx": 4, "ny": 4, "nz": 4}
    constants = {"rho": 1.0, "mu": 0.1, "dt": 0.1, "dx": 0.5, "dy": 0.5, "dz": 0.5}
    fields = {
        "U": np.zeros((5, 4, 4)), 
        "V": np.zeros((4, 5, 4)), 
        "W": np.zeros((4, 4, 5)), 
        "P": np.zeros((4, 4, 4))
    }
    mask_small = np.zeros((2, 2, 2)) 
    
    with pytest.raises(ValueError, match="Spatial Incoherence"):
        assemble_simulation_state(
            config={}, grid=grid, fields=fields, mask=mask_small, 
            constants=constants, boundary_conditions={}, 
            is_fluid=mask_small, is_boundary_cell=mask_small
        )

def test_mask_encapsulation_in_solver_state(dummy_data):
    """Integration: Verifies SolverState correctly holds the 3D mask array."""
    grid = dummy_data["grid"]
    mask_array, _, _ = map_geometry_mask(dummy_data["mask"], grid)
    
    state = SolverState(mask=mask_array, grid=grid)
    assert state.mask.shape == (grid["nx"], grid["ny"], grid["nz"])
    assert np.issubdtype(state.mask.dtype, np.integer)

def test_mask_non_integer_error():
    """Targets map_geometry_mask.py Line 27: Scalar Type Enforcement."""
    from src.step1.map_geometry_mask import map_geometry_mask
    
    # Grid dictionary to satisfy the signature
    grid_ctx = {"nx": 2, "ny": 2, "nz": 2}
    
    # Provide a float (0.5) to trigger the 'isinstance' check on Line 27
    bad_mask = [1, 0, 1, 0, 1, 0, 1, 0.5] 
    
    with pytest.raises(ValueError, match="Mask entries must be finite integers"):
        map_geometry_mask(bad_mask, grid_ctx)

def test_bc_invalid_type_error():
    """Targets parse_boundary_conditions.py Line 32: Invalid BC Type Gate."""
    from src.step1.parse_boundary_conditions import parse_boundary_conditions
    
    # Correct structure: A LIST of dicts, each with a 'location' and 'type'
    bad_bc_list = [
        {
            "location": "x_min", 
            "type": "quantum_flux", # Triggers Line 32
            "values": {}
        }
    ]
    
    dummy_grid_config = {"nx": 2, "ny": 2, "nz": 2}
    
    with pytest.raises(ValueError, match="Invalid boundary type: quantum_flux"):
        parse_boundary_conditions(bad_bc_list, dummy_grid_config)