from typing import Literal, Optional, Union, Any

import zarr
from nibabel import Nifti1Image
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


def _make_compressor(
        name: Union[str, int],
        zarr_version: Literal[2, 3],
        **kwargs: dict
) -> Any:
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


def _load_nifti_from_stream(inp):
    if not hasattr(Nifti1Image, "from_stream"):
        raise Exception("nibabel >=5 is required to read from stream or remote ")
    return Nifti1Image.from_stream(inp)


def _open_zarr(
        out: Union[str, Any],
        mode: Literal["r", "w"] = "w",
        store_opt: Optional[dict] = None,
        **kwargs: dict
) -> Union[zarr.Group, zarr.Array]:
    store_opt = store_opt or {}
    if pyzarr_version == 3:
        StoreLike = zarr.storage.StoreLike
        FsspecStore = zarr.storage.FsspecStore
        LocalStore = zarr.storage.LocalStore
        if "zarr_version" in kwargs:
            kwargs["zarr_format"] = kwargs.pop("zarr_version")
    else:
        StoreLike = zarr.storage.Store
        FsspecStore = zarr.storage.FSStore
        LocalStore = zarr.storage.DirectoryStore
        if "zarr_version" in kwargs or "zarr_format" in kwargs:
            if kwargs.pop("zarr_version", 2) != 2 or kwargs.pop("zarr_format", 2) != 2:
                raise ValueError("Only zarr 2 is supported with zarr-python < 3.0.0")

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


def _create_array(
        out: zarr.Group,
        name: Union[int, str],
        *args: list,
        **kwargs: dict
) -> None:
    """
    Create an array in the given store 'out' with the specified 'name' and options.

    Parameters:
        out : zarr.Group
            The zarr group where the array will be created.
        name :int | str
            The name of the array.

    Raises:
        ValueError: If 'name' is empty.
    """
    if not name:
        raise ValueError("Array name is required")
    name = str(name)

    if "compressor" in kwargs:
        compressor = kwargs.pop("compressor")
    else:
        compressor = kwargs.pop("compressors", None)

    if "dimension_separator" in kwargs and pyzarr_version == 3:
        dimension_separator = kwargs.pop("dimension_separator")
        if dimension_separator == '.' and out.metadata.zarr_format == 2:
            pass
        elif dimension_separator == '/' and out.metadata.zarr_format == 3:
            pass
        else:
            from zarr.core.chunk_key_encodings import ChunkKeyEncodingParams
            dimension_separator = ChunkKeyEncodingParams(
                name="default" if out.metadata.zarr_format == 3 else "v2",
                separator=dimension_separator)

            kwargs["chunk_key_encoding"] = dimension_separator

    if pyzarr_version == 3:
        data = kwargs.pop("data", None)
        out.create_array(name=name, **kwargs, compressors=compressor)
        if data:
            out[name][:] = data
        return
    if pyzarr_version == 2:
        out.create_dataset(name=name, **kwargs, compressor=compressor)
