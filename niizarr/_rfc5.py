import nibabel as nb
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field, asdict as _asdict
from typing import Union
from enum import Enum


NiftiHeader = nb.Nifti1Header | nb.Nifti2Header


class NiftiAxisName(str, Enum):
    dim_0 = x = "x"
    dim_1 = y = "y"
    dim_2 = z = "z"
    dim_3 = t = "t"
    dim_4 = c = "c"
    dim_5 = "dim_5"
    dim_6 = "dim_6"


_UNITS_NIFTI2OME = {
    "unknown": "",
    "meter": "meter",
    "mm": "millimeter",
    "micron": "micrometer",
    "sec": "second",
    "msec": "millisecond",
    "usec": "microsecond",
    "hz": "",
    "ppm": "",
    "rads": "",
}


class NiftiUnit(str, Enum):
    m = meter = "meter"
    mm = millimeter = "millimeter"
    um = micrometer = "micrometer"
    s = second = "second"
    ms = millisecond = "millisecond"
    us = microsecond = "microsecond"

    @classmethod
    def from_nifti(cls, unit: str) -> Union["NiftiUnit", None]:
        unit = _UNITS_NIFTI2OME[unit]
        return NiftiUnit(unit) if unit else None


class NiftiAxisType(str, Enum):
    anatomical = "anatomical"
    space = "space"
    time = "time"
    channel = "channel"

    @classmethod
    def from_name(cls, name: str | NiftiAxisName, anat: bool = True) -> "NiftiAxisType":
        name = NiftiAxisName(name)
        if name in "xyz":
            return cls.anatomical if anat else cls.space
        if name == "t":
            return cls.time
        return cls.channel


class NiftiOrientation(str, Enum):
    R = left_to_right = "left-to-right"
    L = right_to_left = "right-to-left"
    P = anterior_to_posterior = "anterior-to-posterior"
    A = posterior_to_anterior = "posterior-to-anterior"
    S = inferior_to_superior = "inferior-to-superior"
    I = superior_to_inferior = "superior-to-inferior"

    @classmethod
    def from_cosine(cls, vec: tuple[float, float, float]) -> "NiftiOrientation":
        absvec = tuple(map(abs, vec))
        ind = absvec.index(max(absvec))
        sgn = int(vec[ind] > 0)
        return getattr(NiftiOrientation, {0: "LR", 1: "PA", 2: "IS"}[ind][sgn])

    @classmethod
    def from_index(cls, index: int) -> Union["NiftiOrientation", None]:
        return [cls.R, cls.A, cls.S][index] if index < 3 else None


def _dict_factory(data) -> dict:
    return dict(x for x in data if x[1] not in (None, False))


def asdict(data):
    if isinstance(data, (list, tuple)):
        return type(data)(map(asdict, data))
    return _asdict(data, dict_factory=_dict_factory)


@dataclass
class NiftiAxis:
    name: NiftiAxisName
    type: NiftiAxisType
    unit: NiftiUnit | None = None
    orientation: NiftiOrientation | None = None
    discrete: bool = False


class NiftiCoordinateSystemName(str, Enum):
    voxel = "voxel"                 # no unit, same order as array dims
    scaled = "scaled"               # voxel + unit
    anatomical = "anatomical"       # scaled + permutation to RAS+
    scanner = "scanner"             # RAS+
    aligned = "aligned"             # RAS+
    mni = "mni"                     # RAS+
    talairach = "talairach"         # RAS+
    template = "template"           # RAS+
    best = "best"                   # "most likelily to be useful" space

    def from_nifti(name: str) -> Union["NiftiCoordinateSystemName", None]:
        if name == "unknown":
            return None
        return NiftiCoordinateSystemName(name)


@dataclass
class NiftiCoordinateSystem:
    name: NiftiCoordinateSystemName
    axes: list[NiftiAxis]


class CoordinateTransformType(str, Enum):
    identity = "identity"
    mapAxis = "mapAxis"
    translation = "translation"
    scale = "scale"
    affine = "affine"
    rotation = "rotation"
    sequence = "sequence"
    displacements = "displacements"
    coordinates = "coordinates"
    inverseOf = "inverseOf"
    bijection = "bijection"
    byDimension = "byDimension"


@dataclass
class CoordinateTransform:
    type: CoordinateTransformType = field(init=False)
    input: str | None = None
    output: str | None = None


@dataclass
class IdentityTransform(CoordinateTransform):
    type = CoordinateTransformType.identity


@dataclass
class MapAxis(CoordinateTransform):
    type = CoordinateTransformType.mapAxis
    mapAxis: dict[str, str] = field(default_factory=dict)


@dataclass
class Translation(CoordinateTransform):
    type = CoordinateTransformType.translation
    translation: list[float] | None = None
    path: str | None = None


@dataclass
class Scale(CoordinateTransform):
    type: CoordinateTransformType = CoordinateTransformType.scale
    scale: list[float] | None = None
    path: str | None = None


@dataclass
class Affine(CoordinateTransform):
    type = CoordinateTransformType.affine
    affine: list[float] | None = None
    path: str | None = None


@dataclass
class Rotation(CoordinateTransform):
    type = CoordinateTransformType.rotation
    rotation: list[float] | None = None
    path: str | None = None


@dataclass
class Sequence(CoordinateTransform):
    type = CoordinateTransformType.sequence
    transformations: list[CoordinateTransform] = field(default_factory=list)


class Interpolation(str, Enum):
    linear = "linear"
    cubic = "cubic"


@dataclass
class Displacements(CoordinateTransform):
    type = CoordinateTransformType.displacements
    path: str | None = None
    interpolation: Interpolation = Interpolation.linear


@dataclass
class Coordinates(CoordinateTransform):
    type = CoordinateTransformType.coordinates
    path: str | None = None
    interpolation: Interpolation = Interpolation.linear


@dataclass
class InverseOf(CoordinateTransform):
    type = CoordinateTransformType.inverseOf
    transform: CoordinateTransform | None = None


@dataclass
class Bijection(CoordinateTransform):
    type = CoordinateTransformType.bijection
    forward: CoordinateTransform | None = None
    inverse: CoordinateTransform | None = None


@dataclass
class ByDimension(CoordinateTransform):
    type = CoordinateTransformType.byDimension
    transformations: list[CoordinateTransform] = field(default_factory=list)


def get_nifti_axis_names(header: NiftiHeader) -> list[NiftiAxisName]:
    ndim = len(header.get_data_shape())
    names = ["x", "y", "z", "t", "c", "dim_5", "dim_6"][:ndim]
    return [NiftiAxisName(name) for name in names]


def get_cosine_matrix(mat: np.ndarray) -> np.ndarray:
    mat = mat[:3, :3] / ((mat[:3, :3] ** 2).sum(0) ** 0.5)
    mat += np.random.randn(*mat.shape) * 1e-6  # in case 45 deg rotation
    u, _, vh = np.linalg.svd(mat)
    return u @ vh


def nifti2ome(header: NiftiHeader, reverse: bool = False) -> dict:
    """
    Convert a NIfTI header to NGFF metadata using RFC-4 and RFC-5.

    Parameters
    ----------
    header : NiftiHeader
        A NIfTI header read with nibabel.
    reverse : bool, default=False
        If True, the nifti array is converted to a C-order zarr array
        (c, t, y, z, x). If False, the nifti array is converted to a
        F-order zarr array (x, y, z, t, c).
    """

    shape = header.get_data_shape()
    ndim = len(shape)

    qaff, (_, qcode) = header.get_qform(), header.get_qform(coded=True)
    saff, (_, scode) = header.get_sform(), header.get_sform(coded=True)
    qcode = nb.nifti1.xform_codes.label[qcode]
    scode = nb.nifti1.xform_codes.label[scode]
    qcode = NiftiCoordinateSystemName.from_nifti(qcode)
    scode = NiftiCoordinateSystemName.from_nifti(scode)

    if qcode:
        cosine = get_cosine_matrix(qaff)
    elif scode:
        cosine = get_cosine_matrix(saff)
    else:
        cosine = np.eye(3)

    axis_names = get_nifti_axis_names(header)
    axis_types = list(map(NiftiAxisType.from_name, axis_names))
    space_unit, time_unit = map(NiftiUnit.from_nifti, header.get_xyzt_units())

    coordinate_systems = []
    coordinate_transforms = []

    # ------------------------------------------------------------------
    # The unscaled voxel space is identical to the implicit array space,
    # with axes renamed from (dim_0, dim_1, dim_2, dim_3, dim_4) to
    # (x, y, z, t, c).
    coordinate_systems.append(
        unscaled := NiftiCoordinateSystem(
            name=NiftiCoordinateSystemName.voxel,
            axes=[
                NiftiAxis(
                    name=name,
                    type=type,
                    discrete=(type == "channel"),
                    orientation=(
                        NiftiOrientation.from_cosine(cosine[:, i].tolist())
                        if i < 3 else None
                    )
                )
                for i, (name, type) in enumerate(zip(axis_names, axis_types))
            ]
        )
    )

    coordinate_transforms.append(
        IdentityTransform(
            input="/0",
            output=NiftiCoordinateSystemName.voxel
        )
    )

    if reverse:
        unscaled.axes = list(reversed(unscaled.axes))

    # ------------------------------------------------------------------
    # The scaled voxel space is related to the unscaled voxel space,
    # except that voxels now have a physical unit. (0, 0, 0) points to
    # the center of the first voxel.
    coordinate_systems.append(
        scaled := NiftiCoordinateSystem(
            name=NiftiCoordinateSystemName.scaled,
            axes=[
                NiftiAxis(
                    name=name,
                    type=type,
                    unit={"space": space_unit, "time": time_unit}.get(type, None),
                    discrete=(type == "channel"),
                    orientation=(
                        NiftiOrientation.from_cosine(cosine[:, i].tolist())
                        if i < 3 else None
                    )
                )
                for i, (name, type) in enumerate(zip(axis_names, axis_types))
            ]
        )
    )

    zooms = list(map(float, header.get_zooms()))
    coordinate_transforms.append(
        to_scaled := Scale(
            scale=zooms + [1.0] * (ndim - len(zooms)),
            input=NiftiCoordinateSystemName.voxel,
            output=NiftiCoordinateSystemName.scaled,
        )
    )

    if reverse:
        scaled.axes = list(reversed(scaled.axes))
        to_scaled.scale = list(reversed(to_scaled.scale))

    # ------------------------------------------------------------------
    # The anatomical voxel space is the scaled voxel space, with
    # an additional permutation/flip so that (x, y, z) == RAS+
    coordinate_systems.append(
        anatomical := NiftiCoordinateSystem(
            name=NiftiCoordinateSystemName.anatomical,
            axes=[
                NiftiAxis(
                    name=name,
                    type=type,
                    unit={"space": space_unit, "time": time_unit}.get(type, None),
                    discrete=(type == "channel"),
                    orientation=(
                        NiftiOrientation.from_index(i)
                        if i < 3 else None
                    )
                )
                for i, (name, type) in enumerate(zip(axis_names, axis_types))
            ]
        )
    )

    permutation_matrix = np.eye(ndim)
    permutation_matrix[:3, :3] = cosine.round()
    coordinate_transforms.append(
        to_anatomical := Rotation(
            rotation=permutation_matrix[:3].tolist(),
            input=NiftiCoordinateSystemName.scaled,
            output=NiftiCoordinateSystemName.anatomical,
        )
    )

    if reverse:
        scaled.axes = list(reversed(scaled.axes))
        rotation = np.asarray(to_anatomical.rotation)
        rotation[:, :-1] = rotation[:, -2::-1]
        to_anatomical.rotation = rotation.tolist()

    # ------------------------------------------------------------------
    # Finally, the qform and sform can map to one of the following spaces:
    # scanner, aligned, mni, talairach, template

    qform = Rotation(
        rotation=qaff[:3].tolist(),
        input=NiftiCoordinateSystemName.voxel,
        output=NiftiCoordinateSystemName.scanner,
    )

    sform = Affine(
        affine=saff[:3].tolist(),
        input=NiftiCoordinateSystemName.voxel,
        output=NiftiCoordinateSystemName.aligned,
    )

    if reverse:
        rotation = np.asarray(qform.rotation)
        rotation[:, :-1] = rotation[:, -2::-1]
        qform.rotation = rotation.tolist()

        affine = np.asarray(sform.affine)
        affine[:, :-1] = affine[:, -2::-1]
        sform.affine = affine.tolist()

    sanat = qanat = None
    if qcode == scode:
        if not scode:
            # No proper affine, best affine is "anatomical"
            best_space = anatomical
        else:
            sanat = deepcopy(anatomical)
            sanat.name = NiftiCoordinateSystemName(scode)
            coordinate_systems.append(sanat)
            sform.output = sanat.name
            coordinate_transforms.append(sform)
            best_space = sanat
    elif not qcode:
        sanat = deepcopy(anatomical)
        sanat.name = NiftiCoordinateSystemName(scode)
        coordinate_systems.append(sanat)
        sform.output = sanat.name
        coordinate_transforms.append(sform)
        best_space = sanat
    elif not scode:
        qanat = deepcopy(anatomical)
        qanat.name = NiftiCoordinateSystemName(qcode)
        coordinate_systems.append(qanat)
        qform.output = qanat.name
        coordinate_transforms.append(qform)
        best_space = qanat
    else:
        sanat = deepcopy(anatomical)
        sanat.name = NiftiCoordinateSystemName(scode)
        coordinate_systems.append(sanat)
        sform.output = sanat.name
        coordinate_transforms.append(sform)
        qanat = deepcopy(anatomical)
        qanat.name = NiftiCoordinateSystemName(qcode)
        coordinate_systems.append(qanat)
        qform.output = qanat.name
        coordinate_transforms.append(qform)
        if scode < qcode:
            best_space = qanat
        else:
            best_space = sanat

    # ------------------------------------------------------------------
    # "Best" coordinate space
    best = deepcopy(anatomical)
    best.name = NiftiCoordinateSystemName.best
    coordinate_systems.append(best)
    if best_space is anatomical:
        to_best = deepcopy(to_anatomical)
    elif best_space is qanat:
        to_best = deepcopy(qform)
    elif best_space is sanat:
        to_best = deepcopy(sform)
    else:
        assert False
    to_best.output = NiftiCoordinateSystemName.best
    coordinate_transforms.append(to_best)

    # ------------------------------------------------------------------
    # Build JSON
    multiscale = {
        "version": "0.5",
        "coordinateSystems": asdict(coordinate_systems),
        "datasets": [{"path": "0"}],
        "coordinateTransformations": asdict(coordinate_transforms),
    }
    return {"ome": {"multiscales": [multiscale]}}


if __name__ == "__main__":
    import sys
    import json

    fname_nii = sys.argv[1]
    fname_attrs = sys.argv[2]

    header = nb.load(fname_nii).header
    attrs = nifti2ome(header)

    print(json.dumps(attrs, indent=2))

    with open(fname_attrs, "w") as f:
        json.dump(attrs, f, indent=2)
