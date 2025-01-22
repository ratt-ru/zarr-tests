import pytest
import zarr


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
  # inherits from the root level consolidated metadata according to
  # https://zarr.readthedocs.io/en/stable/user-guide/consolidated_metadata.html#usage
  for _, group in root.groups():
    assert (group.metadata.consolidated_metadata is not None) is consolidated



def deconsolidate_metadata(store):
  with open(store / "zarr.json") as f:
    metadata_str = f.read()

  import json

  metadata = json.loads(metadata_str)
  metadata["consolidated_metadata"] = None

  with open(store / "zarr.json", mode="w") as f:
    f.write(json.dumps(metadata))


def test_deconsolidate_metadata(populated_zarr_store):
  # Consolidating metdata at the group level works
  zarr.consolidate_metadata(populated_zarr_store)
  root = zarr.open_group(populated_zarr_store)
  assert root.metadata.consolidated_metadata is not None
  group_names = {n for n, _ in root.groups()}

  # Remove in the root level "zarr.json"
  deconsolidate_metadata(populated_zarr_store)

  root = zarr.open_group(populated_zarr_store)
  assert root.metadata.consolidated_metadata is None
  assert group_names == {n for n, _ in root.groups()}

  for _, g in root.groups():
    assert "DATA" in g
    assert "INDEX" in g