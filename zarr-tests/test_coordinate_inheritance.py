import numpy as np
import xarray
import pytest

@pytest.mark.xfail(reason="child dataset coordinates must align with parent dataset coordinates")
def test_coord_inheritance():

  TIME = 10
  BASELINE = 7*6//2
  FREQUENCY = 16
  POLARIZATION = 4

  data_shape = (TIME, BASELINE, FREQUENCY, POLARIZATION)

  vis_ds = xarray.Dataset({
    "DATA": (("time", "baseline", "frequency", "polarization"), np.random.random(size=data_shape) + np.random.random(size=data_shape)*1j),
    "UVW": (("time", "baseline"), np.random.random(size=(TIME, BASELINE)))
  },
  coords={
    "time": np.arange(TIME),
  })

  FIELD_TIME = 4

  field_and_src_xds = xarray.Dataset({
    "FIELD_PHASE_CENTRE": (("time", "sky_dir_label"), np.random.random((FIELD_TIME, 3)))
  },
  # mismatched coords
  coords={
    "time": np.arange(100, 100 + FIELD_TIME, 1)
  })

  dt = xarray.DataTree.from_dict({
    "vis": vis_ds,
    "vis/field_and_source": field_and_src_xds,
  })

