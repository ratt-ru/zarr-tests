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

@pytest.mark.parametrize("consolidated", [True, False])
def test_consolidated_at_root(populated_zarr_store, consolidated):
  # Consolidate at root level
  if consolidated:
    zarr.consolidate_metadata(populated_zarr_store)

  # Consolidated metadata is available on the root group
  root = zarr.open_group(populated_zarr_store)
  assert (root.metadata.consolidated_metadata is not None) is consolidated


@pytest.mark.parametrize("consolidated", [True, False])
def test_consolidated_at_dataset(populated_zarr_store, consolidated):
  root = zarr.open_group(populated_zarr_store)

  # Consolidate at dataset level
  if consolidated:
    for group_name, _ in root.groups():
      zarr.consolidate_metadata(populated_zarr_store / group_name)

  # Metadata is not consolidated at root level in either case
  root = zarr.open_group(populated_zarr_store)
  assert root.metadata.consolidated_metadata is None

  # But is consolidated at the secondary group (dataset) level
  for _, group in root.groups():
    assert (group.metadata.consolidated_metadata is not None) is consolidated

  # Consolidate at root level
  if consolidated:
    zarr.consolidate_metadata(populated_zarr_store)

  # Root level consolidation reflects
  root = zarr.open_group(populated_zarr_store)
  assert (root.metadata.consolidated_metadata is not None) is consolidated

  # Consolidated metadata at the secondary group (dataset) level
  # still exists but I think is ignored in favour or root level consolidated metadata according
  # https://zarr.readthedocs.io/en/stable/user-guide/consolidated_metadata.html#usage
  for _, group in root.groups():
    assert (group.metadata.consolidated_metadata is not None) is consolidated
