from packaging.version import parse as V
import numpy as np
import zarr
from numpy.lib import NumpyVersion


if V(np.__version__) >= V("2.0"):
    # New code path for newer numpy versions
    def my_function():
        # code using new numpy features
        print("Using new numpy features")
else:
    # Legacy code path
    def my_function():
        # code that works with older numpy versions
        print("Using legacy numpy features")


# If fsspec available, use fsspec
open = open
try:
    import fsspec
    open = fsspec.open
except (ImportError, ModuleNotFoundError):
    fsspec = None


def check_zarr_version(version):
    if version == 3 and V(zarr.__version__) < V("3"):
        raise Exception("zarr version too low")


def _make_compressor(name, zarr_version, **kwargs):
    if not isinstance(name, str):
        return name
    name = name.lower()
    if zarr_version == 2:
        import numcodecs
        compressor_map = {
            "blosc": numcodecs.Blosc,
            "zlib": numcodecs.Zstd,
        }
    elif zarr_version == 3:
        import zarr.codecs
        compressor_map = {
            "blosc": zarr.codecs.BloscCodec,
            "zlib": zarr.codecs.ZstdCodec,
        }
    else:
        raise ValueError(f"zarr version {zarr_version} is not supported")
    if name not in compressor_map:
        raise ValueError('Unknown compressor', name)
    Compressor = compressor_map[name]

    return Compressor(**kwargs)



def _swap_header(header):
    """
    Byteswap the given header array based on the installed numpy version.

    For numpy versions < 2.0.0, it uses header.newbyteorder().
    For numpy versions >= 2.0.0, it uses header.view(header.dtype.newbyteorder()).

    Parameters:
        header (np.ndarray): The numpy array (or structured array) whose byte order needs to be swapped.

    Returns:
        np.ndarray: The array with its byte order swapped.
    
    Note: newbyteorder() does not change data in memory, it only changes how data is interpreted.
          byteswap() changes the data in memory.
    """
    if NumpyVersion(np.__version__) < NumpyVersion("2.0.0"):
        return header.newbyteorder()
    else:
        return header.view(header.dtype.newbyteorder())
