from typing import Literal, Optional, Union, Any

import zarr
import zarr.codecs
from nibabel import Nifti1Image

# If fsspec available, use fsspec
open = open
try:
    import fsspec

    open = fsspec.open
except (ImportError, ModuleNotFoundError):
    fsspec = None

def _make_compressor(name: Union[str, Any], **kwargs: dict) -> Any:
    if not isinstance(name, str):
        return name
    name = name.lower()
    compressor_map = {
        "blosc": zarr.codecs.BloscCodec,
        "zlib": zarr.codecs.ZstdCodec,
    }
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
    if isinstance(out, (zarr.Group, zarr.Array)):
        return out

    if not isinstance(out, (zarr.abc.store.Store, zarr.storage.StorePath)):
        read_only = mode == "r"
        if fsspec:
            storage_options = dict(store_opt)
            protocol = fsspec.core.split_protocol(str(out))[0]
            if protocol in (None, "file", "local"):
                storage_options.setdefault("auto_mkdir", True)
            out = zarr.storage.FsspecStore.from_url(
                out,
                read_only=read_only,
                storage_options=storage_options or None,
            )
        else:
            out = zarr.storage.LocalStore(out, read_only=read_only)
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

    if kwargs.pop("dimension_separator", "/") != "/":
        from zarr.core.chunk_key_encodings import ChunkKeyEncodingParams
        kwargs["chunk_key_encoding"] = ChunkKeyEncodingParams(
            name="default", separator=".")

    if "order" in kwargs:
        kwargs["config"] = {"order": kwargs.pop("order")}

    data = kwargs.pop("data", None)
    out.create_array(name=name, **kwargs, compressors=compressor)
    if data:
        out[name][:] = data
