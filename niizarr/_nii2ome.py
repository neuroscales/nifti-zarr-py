import nibabel as nb
import numpy as np
from typing import List, Tuple, Optional, Union, Mapping
from ._ome import Version

NiftiHeader = Union[nb.Nifti1Header, nb.Nifti2Header]

ORIENTS = {
    "R": "left-to-right",
    "L": "right-to-left",
    "P": "anterior-to-posterior",
    "A": "posterior-to-anterior",
    "S": "inferior-to-superior",
    "I": "superior-to-inferior",
}
RAS_ORIENT = [ORIENTS[c] for c in "RAS"]

UNITS_NIFTI2OME = {
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


class NiftiCoordinateSystemName:
    base = "nifti:base"              # voxel + unit
    anatomical = "nifti:anatomical"  # scaled + permutation to RAS+
    scanner = "nifti:scanner"        # RAS+
    aligned = "nifti:aligned"        # RAS+
    mni = "nifti:mni"                # RAS+
    talairach = "nifti:talairach"    # RAS+
    template = "nifti:template"      # RAS+
    qform = "nifti:qform"            # as defined by qform affine
    sform = "nifti:sform"            # as defined by sform affine

    @classmethod
    def from_nifti(cls, name: str) -> Optional[str]:
        return getattr(cls, name, None)


def _nifti_axis_type_from_name(name: str) -> str:
    if name in "xyz":
        return "space"
    if name == "t":
        return "time"
    return "channel"


def _get_nifti_axis_names(header: NiftiHeader) -> List[str]:
    ndim = len(header.get_data_shape())
    names = ["x", "y", "z", "t", "c", "dim_5", "dim_6"][:ndim]
    return names


def _get_cosine_matrix(mat: np.ndarray) -> np.ndarray:
    mat = mat[:3, :3] / ((mat[:3, :3] ** 2).sum(0) ** 0.5)
    mat += np.random.randn(*mat.shape) * 1e-6  # in case 45 deg rotation
    u, _, vh = np.linalg.svd(mat)
    return u @ vh


def _orient_from_nifti_cosine(vec: Tuple[float, float, float]) -> str:
    absvec = tuple(map(abs, vec))
    ind = absvec.index(max(absvec))
    sgn = int(vec[ind] > 0)
    return ORIENTS.get({0: "LR", 1: "PA", 2: "IS"}[ind][sgn])


def _axis_names_nii2ome(names: List[str], no_time: bool = False) -> List[str]:
    names = names[::-1]                     # reverse axes to follow NGFF spec
    if len(names) == 5:
        names[:2] = reversed(names[:2])     # swap c and t to follow NGFF spec
        if no_time:
            names = names[1:]               # drop time axis if not keeping it
    return names


def _affine_nii2ome(
    aff: np.ndarray,
    ndim: int,
    time_scale: Optional[float] = None,
    time_offset: Optional[float] = None,
) -> np.ndarray:
    ngff_aff = np.eye(ndim + 1)
    ngff_aff[-4:-1, -4:-1] = aff[:3, :3][::-1, ::-1]
    ngff_aff[-4:-1, -1] = aff[:3, -1][::-1]
    if ndim > 3 and time_scale is not None:
        ngff_aff[0, 0] = time_scale
        ngff_aff[0, -1] = time_offset
    return ngff_aff


def nii2ome(
    header: NiftiHeader,
    version: str = "0.6",
    coded_systems: bool = False,
    no_time: bool = False,
) -> dict:
    """
    Convert a NIfTI header to NGFF metadata.

    We choose to name the array axes `([t], [c], [z], y, x)`, following
    the NIfTI convention. Note their permutation, compared to the NIfTI
    order `(x, y, [z, [t, [c]]])`. This is to follow the NGFF specification.

    We choose to name the NGFF intrinsic coordinate system "nifti:base",
    as it corresponds to the NIfTI fallback space when both the qcode
    and scode are set to 0 (unknown).

    If `qcode > 0`, we create a coordinate system named "nifti:qform"
    with the qform affine.

    If `scode > 0`, we create a coordinate system named "nifti:sform"
    with the sform affine.

    If `coded_systems` is True, we also create coordinate systems named after
    the qcode and scode labels, with the same affine as the qform and
    sform coordinate systems respectively. This is to preserve the
    original NIfTI coordinate system names.

    In all cases, the "best" coordinate system is listed last, so that
    it is used by visualization tools by default.

    Parameters
    ----------
    header : NiftiHeader
        A NIfTI header read with nibabel.
    version : str, default="0.6"
        The version of the NGFF specification to use.
    coded_systems : bool, default=False
        Whether to create coordinate systems named after the qcode and
        scode labels.
    no_time : bool, default=False
        Whether to squeeze the time axis if present.

    Returns
    -------
    dict
        A dictionary containing the NGFF metadata for the NIfTI file.
    """
    version = Version(version)
    shape = header.get_data_shape()
    ndim = len(shape)
    nspace = min(3, ndim)

    if ndim > 5:
        raise ValueError(
            f"NIfTI files with more than 5 dimensions are not supported: "
            f"{ndim} > 5"
        )

    # ------------------------------------------------------------------
    # Compute NIfTI spatial transforms (voxel [x,y,z] to world [x,y,z])
    nii_qaff, (_, qcode) = header.get_qform(), header.get_qform(coded=True)
    nii_saff, (_, scode) = header.get_sform(), header.get_sform(coded=True)
    qcode = nb.nifti1.xform_codes.label[qcode]
    scode = nb.nifti1.xform_codes.label[scode]
    qcode = NiftiCoordinateSystemName.from_nifti(qcode)
    scode = NiftiCoordinateSystemName.from_nifti(scode)

    if scode:
        nii_cosine = _get_cosine_matrix(nii_saff)
        nii_orient = list(map(_orient_from_nifti_cosine, nii_cosine.T))
    elif qcode:
        nii_cosine = _get_cosine_matrix(nii_qaff)
        nii_orient = list(map(_orient_from_nifti_cosine, nii_cosine.T))
    else:
        nii_cosine = np.eye(3)
        nii_orient = [None] * 3

    nii_zooms = list(map(float, header.get_zooms()))
    if scode and not qcode:
        svox = (np.sum(nii_saff[:3, :3]**2, axis=0) ** 0.5).tolist()
        for i in range(nspace):
            if nii_zooms[i] == 0:
                nii_zooms[i] = svox[i]
    nii_base = np.eye(4)
    nii_base[:nspace, :nspace] = np.diag(nii_zooms[:nspace])

    # ------------------------------------------------------------------
    # Convert to NGFF affines (scaled [t, c, z, y, x] to world [t, c, z, y, x])
    ngff_ndim = ndim
    if ngff_ndim > 3:
        if no_time:
            ngff_ndim -= 1
            tscale = toffset = None
        else:
            tscale = nii_zooms[3]
            toffset = header["toffset"]
    else:
        tscale = toffset = None

    ngff_base = _affine_nii2ome(nii_base, ngff_ndim, tscale, toffset)
    ngff_qaff = _affine_nii2ome(nii_qaff, ngff_ndim, tscale, toffset)
    ngff_saff = _affine_nii2ome(nii_saff, ngff_ndim, tscale, toffset)
    ngff_qaff = ngff_qaff @ np.linalg.inv(ngff_base)
    ngff_saff = ngff_saff @ np.linalg.inv(ngff_base)

    # ------------------------------------------------------------------
    # Axes
    nii_axes = _get_nifti_axis_names(header)
    axis_names = _axis_names_nii2ome(nii_axes, no_time)
    axis_types = list(map(_nifti_axis_type_from_name, axis_names))
    axis_orient = [None] * (ngff_ndim - nspace) + nii_orient[::-1]
    ras_orient = [None] * (ngff_ndim - nspace) + RAS_ORIENT[:nspace][::-1]
    space_unit, time_unit = map(UNITS_NIFTI2OME.get, header.get_xyzt_units())

    base_axes = [
        dict(
            name=name,
            type=type,
            unit={"space": space_unit, "time": time_unit}.get(type),
            discrete=True if (type == "channel") else None,
            orientation=dict(type="anatomical", value=ort) if ort else None,
        )
        for (name, type, ort) in zip(axis_names, axis_types, axis_orient)
    ]

    ras_axes = [
        dict(
            name=name,
            type=type,
            unit={"space": space_unit, "time": time_unit}.get(type),
            discrete=True if (type == "channel") else None,
            orientation=dict(type="anatomical", value=ort) if ort else None,
        )
        for (name, type, ort) in zip(axis_names, axis_types, ras_orient)
    ]

    coordinate_systems = {}  # ordered dict
    coordinate_transforms = []
    intrinsic_coordinate_transforms = []

    # ------------------------------------------------------------------
    # The intrinsic OME space is a scaled version of the implicit array
    # space, with axes renamed from (dim_0, dim_1, dim_2, dim_3, dim_4)
    # to (t, c, z, y, x) and scaled by the NIfTI zooms.
    # (0, 0, 0) points to the center of the first voxel.
    coordinate_systems[NiftiCoordinateSystemName.base] = dict(
        name=NiftiCoordinateSystemName.base,
        axes=base_axes,
    )

    intrinsic_coordinate_transforms.append(
        dict(
            type="scale",
            scale=np.diag(ngff_base)[:-1].tolist(),
            input="0",
            output=NiftiCoordinateSystemName.base,
        )
    )

    # ------------------------------------------------------------------
    # The anatomical voxel space is the scaled voxel space, with
    # an additional permutation/flip so that (x, y, z) == RAS+
    if coded_systems and (qcode or scode):
        coordinate_systems[NiftiCoordinateSystemName.anatomical] = dict(
            name=NiftiCoordinateSystemName.anatomical,
            axes=ras_axes,
        )

        nii_cosine = nii_cosine[:nspace, :nspace]
        permutation = nii_cosine[::-1, ::-1].round().argmax(0)
        permutation += (ngff_ndim - nspace)  # shift for non-spatial axes
        permutation = permutation.tolist()
        permutation = list(range(ngff_ndim - nspace)) + permutation.tolist()
        coordinate_transforms.append(
            dict(
                type="mapAxis",
                mapAxis=permutation,
                input=NiftiCoordinateSystemName.base,
                output=NiftiCoordinateSystemName.anatomical,
            )
        )

    # ------------------------------------------------------------------
    # Finally, the qform and sform can map to one of the following spaces:
    # scanner, aligned, mni, talairach, template

    if qcode:
        coordinate_systems[NiftiCoordinateSystemName.qform] = dict(
            name=NiftiCoordinateSystemName.qform,
            axes=ras_axes,
        )
        coordinate_transforms.append(
            dict(
                type="rotation",
                rotation=ngff_qaff[:-1].tolist(),
                input=NiftiCoordinateSystemName.base,
                output=NiftiCoordinateSystemName.qform,
            )
        )

        if coded_systems and qcode != scode:
            coordinate_systems[qcode] = dict(
                name=qcode,
                axes=ras_axes,
            )
            coordinate_transforms.append(
                dict(
                    type="identity",
                    input=NiftiCoordinateSystemName.qform,
                    output=qcode,
                )
            )

    if scode:
        coordinate_systems[NiftiCoordinateSystemName.sform] = dict(
            name=NiftiCoordinateSystemName.sform,
            axes=ras_axes,
        )
        coordinate_transforms.append(
            dict(
                type="affine",
                affine=ngff_saff[:-1].tolist(),
                input=NiftiCoordinateSystemName.base,
                output=NiftiCoordinateSystemName.sform,
            )
        )

        if coded_systems:
            coordinate_systems[scode] = dict(
                name=scode,
                axes=ras_axes,
            )
            coordinate_transforms.append(
                dict(
                    type="identity",
                    input=NiftiCoordinateSystemName.sform,
                    output=scode,
                )
            )

    # ------------------------------------------------------------------
    # Build JSON (0.6+RFC4)
    ome = {
        "multiscales": [{
            "coordinateSystems": list(coordinate_systems.values()),
            "coordinateTransformations": coordinate_transforms,
            "datasets": [{
                "path": "0",
                "coordinateTransformations": intrinsic_coordinate_transforms
            }],
        }]
    }
    if version > Version("0.4"):
        ome["version"] = str(version)
    else:
        ome["multiscales"][0]["version"] = str(version)

    # ------------------------------------------------------------------
    # Simplify based on target version

    def simplify(ome: Union[dict, list], simplifier: callable) -> dict:
        if isinstance(ome, list):
            return [simplify(v, simplifier) for v in ome]
        elif isinstance(ome, dict):
            return simplifier({
                k: simplify(v, simplifier) for k, v in ome.items()
            })
        else:
            return ome

    def simplifier(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    if not version.has_rfc(4):
        rfc4_base_simplifier = simplifier

        def simplifier(d: dict) -> dict:
            return rfc4_base_simplifier({
                k: v for k, v in d.items() if k != "orientation"
            })

    if not version.has_rfc(5) and version <= Version("0.5"):
        rfc5_base_simplifier = simplifier

        def simplifier(d: dict) -> dict:
            if "coordinateSystems" in d:
                # in: multiscales
                # > replace "coordinateSystems" with "axes"
                # > remove post-base coordinate transforms
                d = dict(d)
                d["axes"] = d["coordinateSystems"][0]["axes"]
                del d["coordinateSystems"]
                if "coordinateTransformations" in d:
                    del d["coordinateTransformations"]
            elif "coordinateTransformations" in d:
                # in: datasets
                # > remove "input" and "output" from coordinate transforms
                xforms = d["coordinateTransformations"]
                xforms = [dict(xform) for xform in xforms]
                d = dict(d)
                for elem in xforms:
                    del elem["input"]
                    del elem["output"]
                d["coordinateTransformations"] = xforms

            return rfc5_base_simplifier(d)

    return simplify(ome, simplifier)


def ome_add_levels(
    ome: dict,
    scales: List[Mapping[str, float]] = ({"space": 2.0},),
    offsets: List[Mapping[str, float]] = ({"space": 0.5},),
    multiscales_metadata: Optional[dict] = None,
) -> dict:
    """
    Add downsampled levels to OME metadata.

    Parameters
    ----------
    ome : dict
        OME metadata dictionary, with a single multiscale and a single dataset.
    scales : list[dict[str, float]]
        Relative scale of each level compared to the base level.
        Keys in the dictionaries can be axis types ("space", "time")
        or axis names ("x", "y", "z").
    offsets : list[dict[str, float]]
        Relative offset of each level compared to the base level, in
        units of the base level scale. For example, an offset of 0.5
        means that the downsampled grid is shifted by half a voxel
        compared to the base level.
    multiscales_metadata : dict, optional
        Metadata to add to the multiscale level, describing the
        pyramid construction method and parameters.


    Returns
    -------
    ome : dict
        OME metadata dictionary with added levels (modified in-place).
    """
    # Find out OME-NGFF version
    version = ome.get("version") or ome["multiscales"][0].get("version")
    version = Version(version)
    has_coordinate_systems = version.has_rfc(5) or version > Version("0.5")

    # Get base level scale and axes
    multiscale = ome["multiscales"][0]
    level0 = multiscale["datasets"][0]
    scale0 = level0["coordinateTransformations"][0]["scale"]
    if has_coordinate_systems:
        axes = multiscale["coordinateSystems"][0]["axes"]
    else:
        axes = multiscale["axes"]

    def get(scale_or_offset: dict, axis: dict, default: float = 1.0) -> float:
        # Get appropriate scale or offset for the given axis,
        # based on its type or name.
        val = scale_or_offset.get(axis["type"])
        if val is not None:
            return val
        val = scale_or_offset.get(axis["name"])
        if val is not None:
            return val
        return default

    for level in range(1, len(scales) + 1):

        # Compute scale and translation
        scale_level = scales[level - 1]
        if isinstance(scale_level, (list, tuple)):
            scale_level = {
                axis["name"]: s
                for axis, s in zip(axes, scale_level)
            }
        scale = dict(
            type="scale",
            scale=[
                s0 * get(scale_level, axis, 1.0)
                for s0, axis in zip(scale0, axes)
            ]
        )
        if offsets:
            offset_level = offsets[level - 1]
            if isinstance(offset_level, (list, tuple)):
                offset_level = {
                    axis["name"]: s
                    for axis, s in zip(axes, offset_level)
                }
            translation = dict(
                type="translation",
                translation=[
                    s0 * get(offset_level, axis, 0.0)
                    for s0, axis in zip(scale0, axes)
                ]
            )
        else:
            translation = None

        # Generate version-specific transform
        if has_coordinate_systems:
            if translation is not None:
                xform = dict(
                    type="sequence",
                    input=str(level),
                    output="nifti:base",
                    transformations=[scale, translation]
                )
            else:
                xform = scale
                xform["input"] = str(level)
                xform["output"] = "nifti:base"
            xforms = [xform]
        else:
            if translation is not None:
                xforms = [scale, translation]
            else:
                xforms = [scale]

        # Insert new level
        multiscale["datasets"].append({
            "path": str(level),
            "coordinateTransformations": xforms
        })

    if multiscales_metadata:
        if "type" in multiscales_metadata:
            multiscales_metadata = dict(multiscales_metadata)
            multiscale["type"] = multiscales_metadata.pop("type")
        multiscale["metadata"] = multiscales_metadata

    import json
    print(json.dumps(ome, indent=2))
    return ome


if __name__ == "__main__":
    import sys
    import json

    fname_nii = sys.argv[1]
    if len(sys.argv) > 2:
        fname_attrs = sys.argv[2]

    header = nb.load(fname_nii).header
    attrs = nii2ome(header)

    print(json.dumps(attrs, indent=2))

    if len(sys.argv) > 2:
        with open(fname_attrs, "w") as f:
            json.dump(attrs, f, indent=2)
