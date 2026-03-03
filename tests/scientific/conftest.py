import pytest
import numpy as np
import os

@pytest.fixture(scope="session", autouse=True)
def configure_scientific_precision():
    """
    STS Global Configuration:
    Forces high-precision printing and strict numerical error handling 
    for all tests within the tests/scientific/ directory.
    """
    # Force numpy to show full precision in logs/failures
    np.set_printoptions(precision=15, suppress=False, threshold=np.inf)
    
    # Configure global tolerance for STS (1e-12 for double precision)
    os.environ["STS_RTOL"] = "1e-12"
    os.environ["STS_ATOL"] = "1e-15"
    
    yield
    
    # Cleanup (Optional)
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000)

@pytest.fixture
def sts_tolerance():
    """Returns the standard high-precision tolerance for STS assertions."""
    return {"rtol": 1e-12, "atol": 1e-15}
