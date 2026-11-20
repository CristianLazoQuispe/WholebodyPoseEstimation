"""ISLR pose datasets: published HDF5 schema, dataset adapters and the video -> HDF5 builder.

>>> from wholebodypose.datasets import read_hdf5                       # doctest: +SKIP
>>> samples = read_hdf5("ISLR/WLASL/WLASL100/WLASL100_135-Test.hdf5")  # doctest: +SKIP
>>> samples[0]["data"].shape                                            # doctest: +SKIP
(35, 2, 135)

Needs ``pip install 'wholebodypose[datasets]'`` (h5py) for the HDF5 functions.
"""

from wholebodypose.datasets.hdf5 import (
    add_midpoints,
    compare_arrays,
    iter_samples,
    read_hdf5,
    split_filename,
    to_published_array,
    write_sample,
)
from wholebodypose.datasets.sources import SPECS, Sample, load_samples

__all__ = [
    "SPECS",
    "Sample",
    "add_midpoints",
    "compare_arrays",
    "iter_samples",
    "load_samples",
    "read_hdf5",
    "split_filename",
    "to_published_array",
    "write_sample",
]
