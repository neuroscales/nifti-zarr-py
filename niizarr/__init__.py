try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError):
    # `pip install` has not been run.
    # Unset __version__ rather than failing.
    __version__ = None

from ._header import bin2nii  # noqa: F401
from ._nii2zarr import nii2zarr, nii2json, write_nifti_header  # noqa: F401
from ._zarr2nii import zarr2nii, default_nifti_header  # noqa: F401
