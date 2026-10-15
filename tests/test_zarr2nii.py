import os.path as op
import tempfile
import unittest

import nibabel as nib
import numpy as np

import zarr

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
        """OME-Zarr 0.6 metadata reads like the equivalent 0.5 metadata."""
        nifti_file = op.join(DATA, "example4d.nii.gz")
        zarr_file = op.join(self.temp_dir.name, "example4d.nii.zarr")
        nii2zarr(nifti_file, zarr_file, zarr_version=3, ome_version="0.5")
        expected = zarr2nii(zarr_file)

        group = zarr.open_group(zarr_file, mode="r+")
        ome = group.attrs["ome"]
        ms = ome["multiscales"][0]
        datasets = []
        for ds in ms["datasets"]:
            scale, translation = ds["coordinateTransformations"]
            datasets.append({"path": ds["path"], "coordinateTransformations": [{
                "type": "sequence",
                "input": {"path": ds["path"]},
                "output": {"name": "intrinsic"},
                "transformations": [scale, translation],
            }]})
        ms = {k: v for k, v in ms.items() if k != "axes"}
        ms["coordinateSystems"] = [{"name": "intrinsic", "axes": ome["multiscales"][0]["axes"]}]
        ms["datasets"] = datasets
        group.attrs["ome"] = {"version": "0.6", "multiscales": [ms]}

        converted = zarr2nii(zarr_file)
        self.assertEqual(str(expected.header), str(converted.header))
        np.testing.assert_array_equal(expected.get_fdata(), converted.get_fdata())

