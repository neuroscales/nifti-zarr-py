import gzip
import json
import warnings
import os.path as op
import tempfile
import unittest

import nibabel as nib
import numpy as np
import zarr
from packaging.version import parse as V

from niizarr import nii2zarr

try:
    import ome_zarr_models
    VALIDATE = True
except ImportError:
    VALIDATE = False

HERE = op.dirname(op.abspath(__file__))
DATA = op.join(HERE, "data")
OME_VERSIONS = ["0.5", "0.6"]


def _convert(*args, ome_version="0.6", **kwargs):
    if ome_version == "0.5":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return nii2zarr(*args, ome_version=ome_version, **kwargs)
    return nii2zarr(*args, ome_version=ome_version, **kwargs)


class Testnii2zarr(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_non_exist_file(self):
        self.assertRaises(Exception, nii2zarr, 'non_exist_file.nii', self.temp_dir.name)

    def test_input_path(self):
        for ome_version in OME_VERSIONS:
            with self.subTest(ome_version=ome_version):
                _convert(
                    op.join(DATA, "example4d.nii.gz"),
                    self.temp_dir.name,
                    validate=VALIDATE,
                    ome_version=ome_version,
                )

    def test_input_fd(self):
        if V(nib.__version__) < V("5"):
            # nibabel needs to be >= 5 to support from_stream
            return
        for ome_version in OME_VERSIONS:
            with self.subTest(ome_version=ome_version):
                with gzip.open(op.join(DATA, "example4d.nii.gz"), "rb") as f:
                    _convert(
                        f,
                        self.temp_dir.name,
                        validate=VALIDATE,
                        ome_version=ome_version,
                    )

    def test_input_Nifti1Image(self):
        ni = nib.load(op.join(DATA, "example4d.nii.gz"))
        for ome_version in OME_VERSIONS:
            with self.subTest(ome_version=ome_version):
                _convert(
                    ni,
                    self.temp_dir.name,
                    validate=VALIDATE,
                    ome_version=ome_version,
                )

    def test_input_Nifti2Image(self):
        ni = nib.load(op.join(DATA, "example_nifti2.nii.gz"))
        for ome_version in OME_VERSIONS:
            with self.subTest(ome_version=ome_version):
                _convert(
                    ni,
                    self.temp_dir.name,
                    validate=VALIDATE,
                    ome_version=ome_version,
                )

    def test_same_result_nifti1(self):
        written_zarr = op.join(self.temp_dir.name, "example4d.nii.zarr")
        nii2zarr(
            op.join(DATA, "example4d.nii.gz"),
            written_zarr,
            chunk=64,
            validate=VALIDATE,
        )
        written_data = zarr.open(written_zarr)
        for layer in ('0', '1', 'nifti'):
            reference_data = np.load(
                op.join(DATA, "example4d.nii.zarr", f"{layer}.npy"))
            np.testing.assert_array_almost_equal(reference_data, written_data[layer])

    def test_same_result_nifti2(self):
        written_zarr = op.join(self.temp_dir.name, "example_nifti2.nii.zarr")
        nii2zarr(
            op.join(DATA, "example_nifti2.nii.gz"),
            written_zarr,
            chunk=64,
            validate=VALIDATE,
        )
        written_data = zarr.open(written_zarr)
        for layer in ('0', 'nifti'):
            reference_data = np.load(
                op.join(DATA, "example_nifti2.nii.zarr", f"{layer}.npy"))
            np.testing.assert_array_almost_equal(reference_data, written_data[layer])

    def test_zarr_v2_write_unsupported(self):
        with self.assertRaises(ValueError):
            nii2zarr(op.join(DATA, "example4d.nii.gz"), self.temp_dir.name,
                     zarr_version=2)

    def test_ome_v05_future_warning(self):
        with self.assertWarns(FutureWarning):
            nii2zarr(op.join(DATA, "example4d.nii.gz"), self.temp_dir.name,
                     ome_version="0.5")

    def test_named_systems(self):
        """0.6 qform/sform systems map voxels to the NIfTI world coordinates."""
        for name in ("example4d.nii.gz", "example_nifti2.nii.gz"):
            with self.subTest(name=name):
                out = op.join(self.temp_dir.name, name + ".zarr")
                nii2zarr(op.join(DATA, name), out, validate=VALIDATE)
                with open(op.join(out, "zarr.json")) as f:
                    ms = json.load(f)["attributes"]["ome"]["multiscales"][0]
                axes = [a["name"] for a in ms["coordinateSystems"][0]["axes"]]
                scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
                header = nib.load(op.join(DATA, name)).header
                ijk = {"x": 3, "y": 5, "z": 7}
                point = [ijk.get(a, 0) * s for a, s in zip(axes, scale)] + [1]
                for trf in ms["coordinateTransformations"]:
                    kind = trf["output"]["name"].split(":")[1]
                    affine = (header.get_qform() if kind == "qform"
                              else header.get_sform())
                    world = dict(zip(axes, np.asarray(trf["affine"]) @ point))
                    want = affine @ [ijk["x"], ijk["y"], ijk["z"], 1]
                    np.testing.assert_allclose(
                        [world["x"], world["y"], world["z"]], want[:3], atol=1e-6)

    def test_named_systems_absent_in_v05(self):
        out = op.join(self.temp_dir.name, "v05.nii.zarr")
        _convert(op.join(DATA, "example4d.nii.gz"), out, ome_version="0.5")
        with open(op.join(out, "zarr.json")) as f:
            ms = json.load(f)["attributes"]["ome"]["multiscales"][0]
        self.assertNotIn("coordinateSystems", ms)

