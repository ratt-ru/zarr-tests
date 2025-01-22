import icechunk
import numpy as np
import pytest
import xarray




@pytest.fixture
def icechunk_repo(tmp_path):
  store_path = tmp_path / "icechunk-local"
  store_path.mkdir()

  storage_config = icechunk.storage.local_filesystem_storage(store_path)
  repo = icechunk.Repository.create(storage=storage_config)

  yield repo

def test_versioning(icechunk_repo):
  # NOTE(sjperkins): Not working with DataTrees yet
  # dt = xarray.DataTree.from_dict({
  #   "VISIBILITY-01": xarray.Dataset({
  #     "INDEX": (("row",), np.ones(10000, dtype=np.float64)),
  #     "DATA": (("row", "chan", "corr"), np.ones((10000, 64, 4), dtype=np.complex64))
  #   }),
  #   "VISIBILITY-02": xarray.Dataset({
  #     "INDEX": (("row",), np.ones(5000, dtype=np.float64)),
  #     "DATA": (("row", "chan", "corr"), np.ones((5000, 32, 2), dtype=np.complex64))
  #   })
  # })

  ds = xarray.Dataset({
      "INDEX": (("row",), np.ones(10000, dtype=np.float64)),
      "DATA": (("row", "chan", "corr"), np.ones((10000, 64, 4), dtype=np.complex64))
    })

  kw = {"consolidated": False}   # icechunk handles metadata internally

  # Write initial commit
  session = icechunk_repo.writable_session("main")
  ds.to_zarr(session.store, **kw)
  init_commit = session.commit("Initial commit")

  # Append to original dataset
  session = icechunk_repo.writable_session("main")
  ds.to_zarr(session.store, append_dim="row", mode="a", **kw)
  concat_commit = session.commit("Concatenate dataset")
  ds = xarray.open_dataset(session.store, engine="zarr")

  # Add a time column
  session = icechunk_repo.writable_session("main")
  ds = ds.assign(TIME=(("row",), np.linspace(0, 100.0, ds["INDEX"].size)))
  ds.to_zarr(session.store, mode="w", **kw)
  time_commit = session.commit("Add time column")

  # Rollback to initial commit
  session = icechunk_repo.readonly_session(snapshot=init_commit)
  ds = xarray.open_zarr(session.store, **kw)
  assert set(ds.data_vars.keys()) == {"INDEX", "DATA"}
  assert ds.sizes == {"row": 10000, "chan": 64, "corr": 4}

  # Rollback to append commit
  session = icechunk_repo.readonly_session(snapshot=concat_commit)
  ds = xarray.open_zarr(session.store, **kw)
  assert set(ds.data_vars.keys()) == {"INDEX", "DATA"}
  assert ds.sizes == {"row": 20000, "chan": 64, "corr": 4}

  # Rollback to time commit
  session = icechunk_repo.readonly_session(snapshot=time_commit)
  ds = xarray.open_zarr(session.store, **kw)
  assert set(ds.data_vars.keys()) == {"INDEX", "DATA", "TIME"}
  assert ds.sizes == {"row": 20000, "chan": 64, "corr": 4}