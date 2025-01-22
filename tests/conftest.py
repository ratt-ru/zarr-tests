import importlib
import packaging.version

import numpy as np
import pytest
import zarr


ZARR_VERSION = packaging.version.Version(importlib.metadata.version("zarr"))
assert ZARR_VERSION > packaging.version.Version("3.0.0")

@pytest.fixture
def populated_zarr_store(tmp_path):
  """ Constructs a zarr store containing two groups a, b containing some arrays """
  store = tmp_path / "test.zarr"
  root = zarr.create_group(store)
  a = root.create_group("a")
  b = root.create_group("b")

  for g in [a, b]:
    g.attrs["basic"] = {"i": 2, "j": 3.0}
    g.create_array("DATA", shape=(1000, 1000), dtype=np.complex64)
    g.create_array("INDEX", shape=1000, dtype=np.int64)

  yield store