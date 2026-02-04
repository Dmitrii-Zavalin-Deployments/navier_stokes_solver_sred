from dataclasses import dataclass

@dataclass
class GridConfig:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    def __post_init__(self):
        for name, val in [("nx", self.nx), ("ny", self.ny), ("nz", self.nz)]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val}")
