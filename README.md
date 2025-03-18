# Example implementation of the nifti-zarr specification in Python

## Installation

```shell
pip install nifti-zarr
```

## Python API

Convert a nifti file to a nifti-zarr storage.

```python
from niizarr import nii2zarr
nii2zarr("path/to/nifti.nii.gz", "s3://path/to/bucket")
```

Convert a nifti-zarr storage to a nifti file.
The pyramid level can be selected with `level=L`, where 0 is the base/finest level.

```python
from niizarr import zarr2nii
zarr2nii("s3://path/to/bucket", "path/to/nifti.nii.gz", level=0)
```

Load a nifti-zarr into a `nibabel.Nifti1Image` object.

```python
from niizarr import zarr2nii
nivol = zarr2nii("s3://path/to/bucket", level=0)
```

## Command Line Interface

### NIfTI to NIfTI-Zarr

```text
usage: nii2zarr [-h]
                [--chunk CHUNK]
                [--unchunk-channels]
                [--unchunk-time]
                [--levels LEVELS]
                [--method {gaussian,laplacian}]
                [--fill FILL]
                [--compressor {blosc,zlib}]
                [--label]
                [--no-label]
                [--no-time]
                [--no-pyramid-axis {x,y,z}]
                [--zarr-version {2,3}]
                [--ome-version {0.4,0.5}]
                input [output]

Convert nifti to nifti-zarr.

positional arguments:
  input                         Input nifti file.
  output                        Output zarr directory.
                                When not specified, write to input directory.

optional arguments:
  -h, --help                    Show this help message and exit.
  --chunk CHUNK                 Spatial chunk size.
  --unchunk-channels            Save all chanels in a single chunk.
                                Unchunk if you want to display all channels
                                as a single RGB layer in neuroglancer.
                                Chunked by default, unless datatype is RGB.
  --unchunk-time                Save all timepoints in a single chunk.
                                Unchunk if you want to display all timepoints
                                as a single RGB layer in neuroglancer.
                                Chunked by default.
  --levels LEVELS               Number of levels in the pyramid.
                                If -1 (default), use as many levels as possible.
  --method {gaussian,laplacian} Pyramid method.
  --fill FILL                   Missing value.
  --compressor {blosc,zlib}     Compressor.
  --label                       Segmentation volume.
  --no-label                    Not a segmentation volume.
  --no-time                     No time dimension.
  --no-pyramid-axis {x,y,z}     Thick slice axis that should not be downsampled.
  --zarr-version {2,3}          Zarr format version.
  --ome-version {0.4,0.5}       OME-Zarr specification version.
```

### NIfTI-Zarr to NIfTI

```text
usage: zarr2nii [-h] [--level LEVEL] input [output]

Convert nifti-zarr to nifti.

positional arguments:
  input          Input zarr directory
  output         Output nifti file.
                 When not provided, write to the same directory as input

optional arguments:
  -h, --help     Show this help message and exit.
  --level LEVEL  Pyramid level to extract (default: 0 = finest).
```
