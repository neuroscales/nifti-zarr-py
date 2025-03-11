import warnings
from typing import Literal, Optional, Union

import numpy as np
import zarr
from numpy.lib import NumpyVersion
from packaging.version import parse as V

# If fsspec available, use fsspec
open = open
try:
    import fsspec
    open = fsspec.open
except (ImportError, ModuleNotFoundError):
    fsspec = None

if V(zarr.__version__) < V("3"):
    pyzarr_version = 2
else:
    pyzarr_version = 3


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


def _open_zarr(out,
               mode: Literal["r", "w"] = "w",
               store_opt: Optional[dict] = None,
               **kwargs):
    store_opt = store_opt or {}
    if pyzarr_version == 3:
        StoreLike = zarr.storage.StoreLike
        FsspecStore = zarr.storage.FsspecStore
        LocalStore = zarr.storage.LocalStore
    else:
        StoreLike = zarr.storage.Store
        FsspecStore = zarr.storage.FSStore
        LocalStore = zarr.storage.DirectoryStore
        if "zarr_version" in kwargs:
            if kwargs["zarr_version"] != pyzarr_version:
                warnings.warn("zarr_version is ignored in pyzarr version 2")
            kwargs.pop("zarr_version")

    if isinstance(out, (zarr.Group, zarr.Array)):
        return out

    if not isinstance(out, StoreLike):
        if fsspec:
            out = FsspecStore(out, mode=mode, **store_opt)
        else:
            out = LocalStore(out, **store_opt)
    if mode == "w":
        out = zarr.group(store=out, overwrite=True, **kwargs)
    else:
        out = zarr.open(store=out, mode=mode, **kwargs)
    return out


def _create_array(out,
                  name: Union[int, str],
                  *args,
                  **kwargs):
    if not name:
        raise ValueError("Array name is required")
    name = str(name)

    if "compressor" in kwargs:
        compressor = kwargs.pop("compressor")
    else:
        compressor = kwargs.pop("compressors", None)
    if pyzarr_version == 3:
        data = kwargs.pop("data", None)
        out.create_array(name=name, **kwargs, compressors=compressor)
        if data:
            out[name][:] = data
        return
    if pyzarr_version == 2:
        out.create_dataset(name=name, **kwargs, compressor=compressor)
