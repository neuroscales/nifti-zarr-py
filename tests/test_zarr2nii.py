import os.path as op
import tempfile
import unittest

import nibabel as nib
import numpy as np

import warnings

from niizarr import nii2zarr, zarr2nii

HERE = op.dirname(op.abspath(__file__))
DATA = op.join(HERE, "data")


class Testzarr2nii(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_same_result_nifti1(self):
        zarr_file = op.join(DATA, "example4d.nii.zarr")
        nifti_file = op.join(DATA, "example4d.nii.gz")

        converted = zarr2nii(zarr_file)
        loaded = nib.load(nifti_file)

        self.assertEqual(str(loaded.header), str(converted.header))
        self.assertEqual(str(loaded.header.extensions),
                         str(converted.header.extensions))
        np.testing.assert_array_almost_equal(loaded.get_fdata(), converted.get_fdata())

    def test_same_result_nifti2(self):
        zarr_file = op.join(DATA, "example_nifti2.nii.zarr")
        nifti_file = op.join(DATA, "example_nifti2.nii.gz")

        converted = zarr2nii(zarr_file)
        loaded = nib.load(nifti_file)

        self.assertEqual(str(loaded.header), str(converted.header))
        self.assertEqual(str(loaded.header.extensions),
                         str(converted.header.extensions))
        np.testing.assert_array_almost_equal(loaded.get_fdata(), converted.get_fdata())

    def test_ome_v06(self):
        """OME-Zarr 0.6 output reads back like 0.5 output and the source."""
        nifti_file = op.join(DATA, "example4d.nii.gz")
        loaded = nib.load(nifti_file)
        out = {}
        for version in ("0.5", "0.6"):
            out[version] = op.join(self.temp_dir.name, f"v{version}.nii.zarr")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                nii2zarr(nifti_file, out[version], ome_version=version)
        v05, v06 = zarr2nii(out["0.5"]), zarr2nii(out["0.6"])
        self.assertEqual(str(v05.header), str(v06.header))
        np.testing.assert_array_equal(np.asarray(v06.dataobj),
                                      np.asarray(loaded.dataobj))
        self.assertEqual(str(zarr2nii(out["0.5"], level=1).header),
                         str(zarr2nii(out["0.6"], level=1).header))
