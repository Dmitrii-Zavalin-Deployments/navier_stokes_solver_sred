# src/step2/stencil_block.py

from src.common.base_container import ValidatedContainer
from .cell import Cell

class StencilBlock(ValidatedContainer):
    """
    Logical container representing a 7-point stencil + physical context.
    Inherits runtime contract enforcement from ValidatedContainer.
    """

    def __init__(self, center: Cell, i_minus: Cell, i_plus: Cell, 
                 j_minus: Cell, j_plus: Cell, k_minus: Cell, k_plus: Cell,
                 dx: float, dy: float, dz: float, dt: float, 
                 rho: float, mu: float, f_vals: tuple):
        
        # Initialize storage (using private fields)
        self._center = center
        self._i_minus = i_minus
        self._i_plus = i_plus
        self._j_minus = j_minus
        self._j_plus = j_plus
        self._k_minus = k_minus
        self._k_plus = k_plus
        
        self._dx = dx
        self._dy = dy
        self._dz = dz
        self._dt = dt
        self._rho = rho
        self._mu = mu
        self._f_vals = f_vals

    # --- Cell Accessors ---
    @property
    def center(self) -> Cell: return self._get_safe("center")
    @center.setter
    def center(self, val: Cell): self._set_safe("center", val, Cell)

    @property
    def i_minus(self) -> Cell: return self._get_safe("i_minus")
    @i_minus.setter
    def i_minus(self, val: Cell): self._set_safe("i_minus", val, Cell)

    @property
    def i_plus(self) -> Cell: return self._get_safe("i_plus")
    @i_plus.setter
    def i_plus(self, val: Cell): self._set_safe("i_plus", val, Cell)

    @property
    def j_minus(self) -> Cell: return self._get_safe("j_minus")
    @j_minus.setter
    def j_minus(self, val: Cell): self._set_safe("j_minus", val, Cell)

    @property
    def j_plus(self) -> Cell: return self._get_safe("j_plus")
    @j_plus.setter
    def j_plus(self, val: Cell): self._set_safe("j_plus", val, Cell)

    @property
    def k_minus(self) -> Cell: return self._get_safe("k_minus")
    @k_minus.setter
    def k_minus(self, val: Cell): self._set_safe("k_minus", val, Cell)

    @property
    def k_plus(self) -> Cell: return self._get_safe("k_plus")
    @k_plus.setter
    def k_plus(self, val: Cell): self._set_safe("k_plus", val, Cell)

    # --- Physics Parameters Accessors ---
    @property
    def dx(self) -> float: return self._get_safe("dx")
    @dx.setter
    def dx(self, val: float): self._set_safe("dx", val, float)

    @property
    def dy(self) -> float: return self._get_safe("dy")
    @dy.setter
    def dy(self, val: float): self._set_safe("dy", val, float)

    @property
    def dz(self) -> float: return self._get_safe("dz")
    @dz.setter
    def dz(self, val: float): self._set_safe("dz", val, float)

    @property
    def dt(self) -> float: return self._get_safe("dt")
    @dt.setter
    def dt(self, val: float): self._set_safe("dt", val, float)

    @property
    def rho(self) -> float: return self._get_safe("rho")
    @rho.setter
    def rho(self, val: float): self._set_safe("rho", val, float)

    @property
    def mu(self) -> float: return self._get_safe("mu")
    @mu.setter
    def mu(self, val: float): self._set_safe("mu", val, float)

    @property
    def f_vals(self) -> tuple: return self._get_safe("f_vals")
    @f_vals.setter
    def f_vals(self, val: tuple): self._set_safe("f_vals", val, tuple)
