import gzip
import os.path as op
import tempfile
import unittest

import nibabel as nib
import numpy as np
import zarr
from packaging.version import parse as V

from niizarr import nii2zarr
from ._data import compare_zarr_archives

HERE = op.dirname(op.abspath(__file__))
DATA = op.join(HERE, "data")


class Testnii2zarr(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_non_exist_file(self):
        self.assertRaises(Exception, nii2zarr, 'non_exist_file.nii', self.temp_dir.name)

    def test_input_path(self):
        nii2zarr(op.join(DATA, "example4d.nii.gz"), self.temp_dir.name)

    def test_input_fd(self):
        if V(nib.__version__) < V("5"):
            # nibabel needs to be >= 5 to support from_stream
            return
        with gzip.open(op.join(DATA, "example4d.nii.gz"), "rb") as f:
            nii2zarr(f, self.temp_dir.name)

    def test_input_Nifti1Image(self):
        ni = nib.load(op.join(DATA, "example4d.nii.gz"))
        nii2zarr(ni, self.temp_dir.name)

    def test_input_Nifti2Image(self):
        ni = nib.load(op.join(DATA, "example_nifti2.nii.gz"))
        nii2zarr(ni, self.temp_dir.name)

    def test_same_result_nifti1(self):
        written_zarr = op.join(self.temp_dir.name, "example4d.nii.zarr")
        nii2zarr(op.join(DATA, "example4d.nii.gz"), written_zarr, chunk=64)
        self.assertTrue(compare_zarr_archives(written_zarr,
                                              op.join(DATA, "example4d.nii.zarr")))
        written_data = zarr.open(written_zarr)
        for layer in ('0', '1', 'nifti'):
            reference_data = np.load(
                op.join(DATA, "example4d.nii.zarr", f"{layer}.npy"))
            np.testing.assert_array_almost_equal(reference_data, written_data[layer])

    def test_same_result_nifti2(self):
        written_zarr = op.join(self.temp_dir.name, "example_nifti2.nii.zarr")
        nii2zarr(op.join(DATA, "example_nifti2.nii.gz"), written_zarr, chunk=64)
        self.assertTrue(compare_zarr_archives(written_zarr,
                                              op.join(DATA, "example_nifti2.nii.zarr")))
        written_data = zarr.open(written_zarr)
        for layer in ('0', 'nifti'):
            reference_data = np.load(
                op.join(DATA, "example_nifti2.nii.zarr", f"{layer}.npy"))
            np.testing.assert_array_almost_equal(reference_data, written_data[layer])
