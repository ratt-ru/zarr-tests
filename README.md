# Test Repository demonstrating zarr v3 Consolidated Metadata

## Install

Clone the repository and in a virtual environment, pip install in place, then run the test cases:

```bash
virtualenv -p python3.12 ~/venv/zarr-tests
source ~/venv/zarr-tests/bin/activate
(zarr-tests) pip install -e .
(zarr-tests) py.test -s -vvv
```