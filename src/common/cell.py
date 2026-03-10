# src/common/cell.py

from src.common.base_container import ValidatedContainer
import numpy as np

class Cell(ValidatedContainer):
    """
    Lean Topology DTO (Wiring). 
    Holds index, spatial coordinates, mask, and a reference to the global Foundation buffer.
    """
    __slots__ = ['index', 'fields_buffer', 'x', 'y', 'z', 'mask', 'is_ghost']

    def __init__(self, index: int, fields_buffer: np.ndarray, x: int, y: int, z: int, mask: int, is_ghost: bool = False):
        # Topology data (stays in the object)
        super().__setattr__('index', index)
        super().__setattr__('fields_buffer', fields_buffer)
        super().__setattr__('x', x)
        super().__setattr__('y', y)
        super().__setattr__('z', z)
        super().__setattr__('mask', mask)
        super().__setattr__('is_ghost', is_ghost)

    # --- Physical Fields (View into Foundation) ---
    @property
    def vx(self) -> float: return self.fields_buffer[self.index, 0]
    @vx.setter
    def vx(self, value: float): self.fields_buffer[self.index, 0] = value

    @property
    def vy(self) -> float: return self.fields_buffer[self.index, 1]
    @vy.setter
    def vy(self, value: float): self.fields_buffer[self.index, 1] = value

    @property
    def vz(self) -> float: return self.fields_buffer[self.index, 2]
    @vz.setter
    def vz(self, value: float): self.fields_buffer[self.index, 2] = value

    @property
    def vx_star(self) -> float: return self.fields_buffer[self.index, 3]
    @vx_star.setter
    def vx_star(self, value: float): self.fields_buffer[self.index, 3] = value

    @property
    def vy_star(self) -> float: return self.fields_buffer[self.index, 4]
    @vy_star.setter
    def vy_star(self, value: float): self.fields_buffer[self.index, 4] = value

    @property
    def vz_star(self) -> float: return self.fields_buffer[self.index, 5]
    @vz_star.setter
    def vz_star(self, value: float): self.fields_buffer[self.index, 5] = value

    @property
    def p(self) -> float: return self.fields_buffer[self.index, 6]
    @p.setter
    def p(self, value: float): self.fields_buffer[self.index, 6] = value

    @property
    def p_next(self) -> float: return self.fields_buffer[self.index, 7]
    @p_next.setter
    def p_next(self, value: float): self.fields_buffer[self.index, 7] = value