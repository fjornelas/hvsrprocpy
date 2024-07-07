import pytest
import numpy as np
import os
from hvsrprocpy.tdt import *
from hvsrprocpy.fdt import *

# Define mock input data for testing
# Example data, replace with actual data if available
h1 = np.random.rand(1000)
h2 = np.random.rand(1000)
v = np.random.rand(1000)
dt = 0.01
time_ts = np.arange(0, 10, dt)
output_dir = './test_output'

@pytest.fixture(scope="module")
def setup_teardown():
    # Setup code
    print("\nSetup before running tests")
    os.makedirs(output_dir, exist_ok=True)

    yield

    # Teardown code
    print("\nTeardown after running tests")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(output_dir)

def test_hvsr_function(setup_teardown):
    # Test case 1: Default parameters
    hvsr(h1, h2, v, dt, time_ts, output_dir)

    # Test case 2: Custom parameters
    custom_kwargs = {
        'plot_ts': False,
        'plot_hvsr': False,
        'output_metadata': False,
        'output_removed_hvsr': True,
        'output_polar_curves': True
    }
    hvsr(h1, h2, v, dt, time_ts, output_dir, **custom_kwargs)

    # Check if the output files are generated
    assert os.path.exists(os.path.join(output_dir, 'hvsr_unsel.csv'))
    assert os.path.exists(os.path.join(output_dir, 'hvsr_polar.csv'))

    # Test case 3: Edge case where no windows are selected
    with pytest.raises(ValueError):
        hvsr(h1, h2, v, dt, time_ts, output_dir, win_width=2000)

    # Additional test cases can be added as needed

if __name__ == '__main__':
    pytest.main()
