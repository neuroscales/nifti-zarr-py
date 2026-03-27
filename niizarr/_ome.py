from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Tuple, Optional, Union
from warnings import warn
from numbers import Real


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------


class ValidationError(ValueError):
    pass


class ValidationWarning(Warning):
    pass


def validate(condition: bool, message: str, strict: bool = False) -> None:
    if not condition:
        if strict:
            raise ValidationError(message)
        else:
            warn(message, ValidationWarning)


# ----------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------


VersionMajorType = int
VersionMinorType = int
VersionPatchType = Tuple[str, Optional[int]]  # (prefix, suffix)
RFCsType = Tuple[int, ...]
VersionTupleType = Tuple[
    VersionMajorType, VersionMinorType, VersionPatchType, RFCsType
]
PATCH_PREFIX_ORDER = ('dev', 'alpha', 'beta', 'rc', '')


def _version_tuple(version: str) -> VersionTupleType:
    version, *rfcs = version.split("+")
    if any(not x.startswith("rfc") or not x[3:].isdigit() for x in rfcs):
        raise ValueError(f"Invalid RFC version: {version}+{'+'.join(rfcs)}")
    rfcs = tuple(int(x[3:]) for x in rfcs)
    major, minor, *patch = tuple(version.split("."))
    major, minor = int(major), int(minor)
    patch = patch[0] if patch else None
    if patch:
        patch = _split_patch(patch)
    else:
        patch = '', None
    return (major, minor, patch, rfcs)


def _split_patch(patch: str) -> VersionPatchType:
    for prefix in PATCH_PREFIX_ORDER:
        if patch.startswith(prefix):
            suffix = patch[len(prefix):]
            suffix = int(suffix) if suffix.isdigit() else None
            return prefix, suffix
    raise ValueError(f"Invalid patch version: {patch}")


def _patch_gt(patch: VersionPatchType, other: VersionPatchType) -> bool:
    if PATCH_PREFIX_ORDER.index(patch[0]) > PATCH_PREFIX_ORDER.index(other[0]):
        return True
    if PATCH_PREFIX_ORDER.index(patch[0]) < PATCH_PREFIX_ORDER.index(other[0]):
        return False
    if patch[1] is None:
        return False
    if other[1] is None:
        return True
    return patch[1] > other[1]


class Version(str):
    """
    Parses OME version strings of the form:
    - {MAJOR}.{MINOR}
    - {MAJOR}.{MINOR}.{PATCH}
    - {MAJOR}.{MINOR}.{PRERELEASE}{PATCH}
    - {MAJOR}.{MINOR}+rfc{RFC}
    """

    @property
    def as_tuple(self) -> VersionTupleType:
        return _version_tuple(self)

    def has_rfc(self, rfc: int) -> bool:
        return rfc in self.as_tuple[3]

    def __eq__(self, other: str) -> bool:
        return self.as_tuple == Version(other).as_tuple

    def __gt__(self, other: str) -> bool:
        other = Version(other)
        if self.as_tuple[:2] > other.as_tuple[:2]:
            return True
        if self.as_tuple[:2] < other.as_tuple[:2]:
            return False
        return _patch_gt(self.as_tuple[2], other.as_tuple[2])

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return f"Version('{str(self)}')"


# ----------------------------------------------------------------------
# Known enumerations
# ----------------------------------------------------------------------


class MyStrEnumFactory(type):

    def __new__(cls, name, bases, namespace, /, **kwds):
        kls = super().__new__(cls, name, bases, namespace, **kwds)
        SINGLETONS = {}
        MEMBERS = {}

        def unroll_hierarchy(cls) -> None:
            for key, value in cls.__dict__.items():
                if key[:2] == "__":
                    continue
                if isinstance(value, str):
                    if value not in SINGLETONS:
                        SINGLETONS[value] = str.__new__(kls, value)
                        MEMBERS[value] = SINGLETONS[value]
                    MEMBERS[key] = SINGLETONS[value]
            for key, value in kls.__dict__.items():
                if key == "_ALIASES":
                    for alias, target in value.items():
                        if target not in SINGLETONS:
                            raise ValueError(
                                f"Alias target {target} is not a valid member"
                            )
                        MEMBERS[alias] = SINGLETONS[target]
            for base in cls.__bases__:
                if isinstance(base, MyStrEnumFactory):
                    unroll_hierarchy(base)

        unroll_hierarchy(kls)
        kls._SINGLETONS = SINGLETONS
        kls._MEMBERS = MEMBERS
        for key, value in MEMBERS.items():
            setattr(kls, key, value)
        return kls

    def __contains__(cls, value: str) -> bool:
        return value in cls._SINGLETONS


class MyStrEnum(str, metaclass=MyStrEnumFactory):

    _ALIASES = {}
    _MEMBERS = {}

    def __new__(cls, value: str) -> "MyStrEnum":
        if value in cls._SINGLETONS:
            return cls._SINGLETONS[value]
        raise ValueError(f"{value} is not a valid {cls.__name__}")

    @classmethod
    def from_str(cls, value: str) -> Optional["TimeUnit"]:
        return getattr(cls, cls._ALIASES.get(value),  None)

    @classmethod
    def has(cls, value: str, strict: bool = True) -> bool:
        if strict:
            try:
                TimeUnit(value)
                return True
            except ValueError:
                return False
        else:
            return cls.from_str(value) is not None


# --- Units ------------------------------------------------------------


class SpaceUnit(MyStrEnum):
    angstrom = "angstrom"
    am = attometer = "attometer"
    cm = centimeter = "centimeter"
    dm = decimeter = "decimeter"
    Em = exameter = "exameter"
    fm = femtometer = "femtometer"
    ft = foot = "foot"
    Gm = gigameter = "gigameter"
    Hm = hm = hectometer = "hectometer"
    inch = "inch"
    Km = km = kilometer = "kilometer"
    Mm = megameter = "megameter"
    m = meter = "meter"
    um = micrometer = "micrometer"
    mi = mile = "mile"
    mm = millimeter = "millimeter"
    nm = nanometer = "nanometer"
    pc = parsec = "parsec"
    Pm = petameter = "petameter"
    pm = picometer = "picometer"
    Tm = terameter = "terameter"
    yd = yard = "yard"
    ym = yoctometer = "yoctometer"
    Ym = yottameter = "yottameter"
    zm = zeptometer = "zeptometer"
    Zm = zettameter = "zettameter"

    _ALIASES = {
        "Å": "angstrom",
        "in": "inch",
        "μm": "micrometer",
    }


class TimeUnit(MyStrEnum):
    attosecond = "attosecond"
    cs = centisecond = "centisecond"
    d = day = "day"
    ds = decisecond = "decisecond"
    Es = exasecond = "exasecond"
    fs = femtosecond = "femtosecond"
    Gs = gigasecond = "gigasecond"
    Hs = hs = hectosecond = "hectosecond"
    H = h = hour = "hour"
    Ks = ks = kilosecond = "kilosecond"
    Ms = megasecond = "megasecond"
    us = microsecond = "microsecond"
    ms = millisecond = "millisecond"
    minute = "minute"
    ns = nanosecond = "nanosecond"
    Ps = petasecond = "petasecond"
    ps = picosecond = "picosecond"
    s = second = "second"
    Ts = terasecond = "terasecond"
    ys = yoctosecond = "yoctosecond"
    Ys = yottasecond = "yottasecond"
    zs = zeptosecond = "zeptosecond"
    Zs = zettasecond = "zettasecond"

    _ALIASES = {
        "as": "attosecond",
        "μs": "microsecond",
    }


class Unit(SpaceUnit, TimeUnit):

    @classmethod
    def from_nifti(cls, unit: str) -> Optional["Unit"]:
        unit = _UNITS_NIFTI2OME[unit]
        return Unit(unit) if unit else None


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


# --- Axes ------------------------------------------------------------


class AxisType(MyStrEnum):
    array = "array"
    space = "space"
    time = "time"
    channel = "channel"
    coordinate = "coordinate"
    displacement = "displacement"


# --- Orientation ------------------------------------------------------


class BaseOrientationValue(MyStrEnum):
    R = LR = left_to_right = "left-to-right"
    L = RL = right_to_left = "right-to-left"
    proximal_to_distal = "proximal-to-distal"
    distal_to_proximal = "distal-to-proximal"


class BipedOrientationValue(BaseOrientationValue):
    P = AP = anterior_to_posterior = front_to_back = "anterior-to-posterior"
    A = PA = posterior_to_anterior = back_to_front = "posterior-to-anterior"
    S = IS = inferior_to_superior = feet_to_head = "inferior-to-superior"
    I = SI = superior_to_inferior = head_to_feet = "superior-to-inferior"  # noqa: E501, E741
    dorsal_to_palmar = back_of_hand_to_palm = "dorsal-to-palmar"
    palmar_to_dorsal = palm_to_back_of_hand = "palmar-to-dorsal"
    dorsal_to_plantar = top_of_foot_to_sole = "dorsal-to-plantar"
    plantar_to_dorsal = sole_to_top_of_foot = "plantar-to-dorsal"


class QuadripedOrientationValue(BaseOrientationValue):
    rostral_to_caudal = nose_to_tail = beak_to_tail = "rostral-to-caudal"
    caudal_to_rostral = tail_to_nose = tail_to_beak = "caudal-to-rostral"
    cranial_to_caudal = head_to_tail = "cranial-to-caudal"
    caudal_to_cranial = tail_to_head = "caudal-to-cranial"
    dorsal_to_ventral = back_to_belly = top_to_bottom = "dorsal-to-ventral"
    ventral_to_dorsal = belly_to_back = bottom_to_top = "ventral-to-dorsal"


class OrientationValue(BipedOrientationValue, QuadripedOrientationValue):

    @classmethod
    def from_nifti_cosine(
        cls, vec: Tuple[float, float, float]
    ) -> "OrientationValue":
        absvec = tuple(map(abs, vec))
        ind = absvec.index(max(absvec))
        sgn = int(vec[ind] > 0)
        return getattr(OrientationValue, {0: "LR", 1: "PA", 2: "IS"}[ind][sgn])

    @classmethod
    def _orient_from_index(cls, index: int) -> Optional["OrientationValue"]:
        return [cls.R, cls.A, cls.S][index] if index < 3 else None


class OrientationType(MyStrEnum):
    anatomical = "anatomical"


# --- Schema base ------------------------------------------------------

def _is_strenum(v):
    return isinstance(v, MyStrEnum)


class JsonSerializable:

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict) -> "JsonSerializable":
        raise NotImplementedError

    def validate(self, version: Version, strict: bool = False) -> None:
        return


@dataclass
class DictSerializable(JsonSerializable):

    @classmethod
    def _dict_factory(cls, items):
        return {
            k: (
                v.to_json() if isinstance(v, DictSerializable) else
                str(v) if _is_strenum(v) else
                v
            )
            for k, v in items
            if v is not None
        }

    def to_json(self) -> dict:
        return asdict(self, dict_factory=self._dict_factory)


class ListSerializable(list, JsonSerializable):

    def to_json(self) -> dict:
        return [
            item.to_json() if isinstance(item, JsonSerializable) else
            str(item) if _is_strenum(item) else
            item
            for item in self
        ]


# --- Schema -----------------------------------------------------------


@dataclass
class Orientation(DictSerializable):
    type: OrientationType = OrientationType.anatomical
    value: Optional[OrientationValue] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "Axis":
        if data is None:
            return None
        return cls(
            type=OrientationType(data.get("type")),
            value=OrientationValue(data.get("value"))
        )

    def validate(self, version: Version, strict: bool = False) -> None:
        validate(
            self.type == OrientationType.anatomical,
            f"Unknown orientation type: {self.type}\n{self}",
            strict=True
        )


@dataclass
class Axis(DictSerializable):
    name: Optional[str] = None
    type: Optional[Union[AxisType, str]] = None
    unit: Optional[Union[Unit, str]] = None
    discrete: Optional[bool] = None
    longName: Optional[str] = None
    orientation: Optional[Orientation] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "Axis":
        if data is None:
            return None
        try:
            type = AxisType(data["type"])
        except ValueError:
            type = data.get("type")
        try:
            unit = Unit(data["unit"])
        except ValueError:
            unit = data.get("unit")
        return cls(
            name=data.get("name"),
            type=type,
            unit=unit,
            discrete=data.get("discrete"),
            longName=data.get("longName"),
            orientation=Orientation.from_json(data.get("orientation"))
        )

    def validate(self, version: Version, strict: bool = False) -> None:
        validate(
            isinstance(self.name, str),
            "Axis name MUST be set",
            strict=True
        )
        if self.type == AxisType.space:
            validate(
                self.unit in SpaceUnit,
                f"Space axis unit SHOULD be a valid space unit: {self.unit}"
                f"\n{self}",
                strict=strict)
        elif self.type == AxisType.time:
            validate(
                self.unit in TimeUnit,
                f"Time axis unit SHOULD be a valid time unit: {self.unit}"
                f"\n{self}",
                strict=strict)
        elif self.type in (None, AxisType.channel):
            validate(
                self.unit is None,
                f"Channel axis unit SHOULD NOT be set\n{self}",
                strict=strict)
        else:
            validate(
                False,
                f"Unknown axis type {self.type}\n{self}",
                strict=False
            )
            validate(
                self.unit is None,
                f"Custom axis unit SHOULD NOT be set\n{self}",
                strict=strict)


class Axes(ListSerializable):

    @classmethod
    def from_json(cls, data: Optional[List[dict]]) -> "Axes":
        if data is None:
            return None
        return cls(Axis.from_json(item) for item in data)

    def validate(self, version: Version, strict: bool = False) -> None:
        for axis in self:
            axis.validate(version, strict)


@dataclass
class CoordinateSystem(DictSerializable):
    name: str
    axes: Axes

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "CoordinateSystem":
        if data is None:
            return None
        return cls(
            name=data.get("name"),
            axes=Axes.from_json(data.get("axes"))
        )

    def validate(self, version: Version, strict: bool = False) -> None:
        validate(
            isinstance(self.name, str),
            f"CoordinateSystem name must be a string\n{self}",
            strict=True
        )
        self.axes.validate(version, strict)


@dataclass
class IntrinsicCoordinateSystem(CoordinateSystem):

    def validate(self, version: Version, strict: bool = False) -> None:
        super().validate(version, strict)
        validate(
            3 <= len(self.axes) <= 5,
            "Intrinsic coordinate system must have between 3 and 5 axes",
            strict=True
        )
        space_axes = []
        ntime = nchannel = nspace = 0
        for axis in self.axes:
            if axis.type == AxisType.time:
                validate(
                    ntime == 0,
                    f"Intrinsic coordinate system cannot have multiple "
                    f"time axes\n{self}",
                    strict=True
                )
                self.validate(
                    nchannel == 0 and nspace == 0,
                    f"Time axis must come before channel and space axes"
                    f"\n{self}",
                    strict=True
                )
                ntime += 1
            elif axis.type not in (AxisType.time, AxisType.space):
                validate(
                    nchannel == 0,
                    f"Intrinsic coordinate system cannot have multiple "
                    f"channel axes\n{self}",
                    strict=True
                )
                validate(
                    nspace == 0,
                    f"Channel axis must come before space axes\n{self}",
                    strict=True
                )
                nchannel += 1
            elif axis.type == AxisType.space:
                validate(
                    nspace < 3,
                    f"Intrinsic coordinate system cannot have more than "
                    f"3 space axes\n{self}",
                    strict=True
                )
                nspace += 1
                space_axes.append(axis.name)
        validate(
            nspace > 1,
            f"Intrinsic coordinate system must have at least 2 space axes"
            f"\n{self}",
            strict=True
        )
        space_axes = tuple(space_axes)
        recommended_space_axes = (
            ("y", "x"),
            ("z", "y", "x")
        )
        validate(
            space_axes in recommended_space_axes,
            f"Recommended space axis names are {recommended_space_axes}"
            f"\n{self}",
            strict=strict
        )


class CoordinateSystems(ListSerializable):

    @classmethod
    def from_json(cls, data: Optional[List[dict]]) -> "CoordinateSystems":
        if data is None:
            return None
        return cls(CoordinateSystem.from_json(item) for item in data)

    def validate(self, version: Version, strict: bool = False) -> None:
        for cs in self:
            cs.validate(version, strict)


class CoordinateTransformationType(str, Enum):
    identity = "identity"
    mapAxis = "mapAxis"
    translation = "translation"
    scale = "scale"
    affine = "affine"
    rotation = "rotation"
    sequence = "sequence"
    displacements = "displacements"
    coordinates = "coordinates"
    bijection = "bijection"
    byDimension = "byDimension"


@dataclass
class CoordinateTransformation(DictSerializable):
    type: Optional[CoordinateTransformationType] = None
    name: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    input_axes: Optional[List[int]] = None
    output_axes: Optional[List[int]] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "CoordinateTransformation":
        if data is None:
            return None
        cls = {
            "identity": IdentityTransformation,
            "mapAxis": MapAxisTransformation,
            "translation": TranslationTransformation,
            "scale": ScaleTransformation,
            "affine": AffineTransformation,
            "rotation": RotationTransformation,
            "sequence": SequenceTransformation,
            "displacements": DisplacementsTransformation,
            "coordinates": CoordinatesTransformation,
            "bijection": BijectionTransformation,
            "byDimension": ByDimensionTransformation,
        }[data["type"]]
        return cls.from_json(data)

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        if version > Version("0.5") and not wrapped:
            validate(
                self.type in (
                    SequenceTransformation,
                    BijectionTransformation,
                    ByDimensionTransformation,
                ) or (self.input is not None and self.output is not None),
                "(>=0.6) CoordinateTransformation must have both input "
                "and output."
                f"\n{self}",
                strict=True
            )


class CoordinateTransformations(ListSerializable):

    @classmethod
    def from_json(
        cls, data: Optional[List[dict]]
    ) -> "CoordinateTransformations":
        if data is None:
            return None
        return cls(CoordinateTransformation.from_json(item) for item in data)

    def validate(self, version: Version, strict: bool = False) -> None:
        for ct in self:
            ct.validate(version, strict)


@dataclass
class IdentityTransformation(CoordinateTransformation):
    type: CoordinateTransformationType = CoordinateTransformationType.identity

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "IdentityTransformation":
        if data is None:
            return None
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.identity,
            f"Invalid type {self.type} for IdentityTransformation\n{self}",
            strict=True
        )


@dataclass
class MapAxisTransformation(CoordinateTransformation):
    type: CoordinateTransformationType = CoordinateTransformationType.mapAxis
    mapAxis: Optional[List[int]] = None

    @classmethod
    def from_json(cls, data: dict) -> "MapAxisTransformation":
        return cls(
            type=CoordinateTransformationType(data["type"]),
            mapAxis=data["mapAxis"],
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.mapAxis,
            f"Invalid type {self.type} for MapAxisTransformation\n{self}",
            strict=True
        )
        validate(
            isinstance(self.mapAxis, list) and len(self.mapAxis) > 0,
            f"mapAxis transformation must have a non-empty mapAxis list"
            f"\n{self}",
            strict=True
        )
        validate(
            all(isinstance(i, int) and i >= 0 for i in self.mapAxis),
            f"All elements of mapAxis must be non-negative integers"
            f"\n{self}",
            strict=True
        )


@dataclass
class TranslationTransformation(CoordinateTransformation):
    type: CoordinateTransformationType \
        = CoordinateTransformationType.translation
    translation: Optional[List[float]] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "TranslationTransformation":
        if data is None:
            return None
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            translation=data.get("translation"),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.translation,
            f"Invalid type {self.type} for TranslationTransformation",
            strict=True
        )
        validate(
            isinstance(self.translation, list) and len(self.translation) > 0,
            f"translation transformation must have a non-empty translation "
            f"list\n{self}",
            strict=True
        )
        validate(
            all(isinstance(i, Real) for i in self.translation),
            f"All elements of translation must be numbers\n{self}",
            strict=True
        )


@dataclass
class ScaleTransformation(CoordinateTransformation):
    type: CoordinateTransformationType \
        = CoordinateTransformationType.scale
    scale: Optional[List[float]] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "ScaleTransformation":
        if data is None:
            return None
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            scale=data.get("scale"),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.scale,
            f"Invalid type {self.type} for ScaleTransformation",
            strict=True
        )
        validate(
            isinstance(self.scale, list) and len(self.scale) > 0,
            f"scale transformation must have a non-empty scale list\n{self}",
            strict=True
        )
        validate(
            all(isinstance(i, Real) for i in self.scale),
            f"All elements of scale must be numbers\n{self}",
            strict=True
        )


@dataclass
class AffineTransformation(CoordinateTransformation):
    type: CoordinateTransformationType \
        = CoordinateTransformationType.affine
    affine: Optional[List[List[float]]] = None
    path: Optional[str] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "AffineTransformation":
        if data is None:
            return None
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            affine=data.get("affine"),
            path=data.get("path"),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.affine,
            f"Invalid type {self.type} for AffineTransformation",
            strict=True
        )
        validate(
            (self.affine is None) != (self.path is None),
            "AffineTransformation must have either affine or path",
            strict=True
        )
        if self.affine is not None:
            validate(
                isinstance(self.affine, list) and len(self.affine) > 0,
                f"affine transformation must have a non-empty affine list"
                f"\n{self}",
                strict=True
            )
            validate(
                all(isinstance(row, list) for row in self.affine) and
                all(isinstance(i, Real) for row in self.affine for i in row),
                f"All elements of affine must be lists of numbers\n{self}",
                strict=True
            )


@dataclass
class RotationTransformation(CoordinateTransformation):
    type: CoordinateTransformationType \
        = CoordinateTransformationType.rotation
    rotation: Optional[List[List[float]]] = None
    path: Optional[str] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "RotationTransformation":
        if data is None:
            return None
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            rotation=data.get("rotation"),
            path=data.get("path"),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.rotation,
            f"Invalid type {self.type} for RotationTransformation",
            strict=True
        )
        validate(
            (self.rotation is None) != (self.path is None),
            "RotationTransformation must have either rotation or path",
            strict=True
        )
        if self.rotation is not None:
            validate(
                isinstance(self.rotation, list) and len(self.rotation) > 0,
                f"rotation transformation must have a non-empty rotation "
                f"list\n{self}",
                strict=True
            )
            self.validate(
                all(isinstance(row, list) for row in self.rotation) and
                all(isinstance(i, Real) for row in self.rotation for i in row),
                f"All elements of rotation must be lists of numbers"
                f"\n{self}",
                strict=True
            )


@dataclass
class SequenceTransformation(CoordinateTransformation):
    type: CoordinateTransformationType \
        = CoordinateTransformationType.sequence
    transformations: Optional[CoordinateTransformations] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "SequenceTransformation":
        if data is None:
            return None
        Xforms = CoordinateTransformations
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            transformations=Xforms.from_json(data.get("transformations")),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.sequence,
            f"Invalid type {self.type} for SequenceTransformation\n{self}",
            strict=True
        )
        validate(
            self.transformations is not None and len(self.transformations) > 0,
            f"sequence must have a non-empty transformations list\n{self}",
            strict=True
        )
        for item in self.transformations:
            item.validate(version, strict, wrapped=True)


@dataclass
class DisplacementsTransformation(CoordinateTransformation):
    type: CoordinateTransformationType \
        = CoordinateTransformationType.displacements
    path: Optional[str] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "DisplacementsTransformation":
        if data is None:
            return None
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            path=data.get("path"),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.displacements,
            f"Invalid type {self.type} for DisplacementsTransformation"
            f"\n{self}",
            strict=True
        )
        validate(
            isinstance(self.path, str),
            f"displacements path must be a string\n{self}",
            strict=True
        )


@dataclass
class CoordinatesTransformation(CoordinateTransformation):
    type: CoordinateTransformationType \
        = CoordinateTransformationType.coordinates
    path: Optional[str] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "CoordinatesTransformation":
        if data is None:
            return None
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            path=data.get("path"),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.coordinates,
            f"Invalid type {self.type} for CoordinatesTransformation"
            f"\n{self}",
            strict=True
        )
        validate(
            isinstance(self.path, str),
            f"coordinates path must be a string\n{self}",
            strict=True
        )


@dataclass
class BijectionTransformation(CoordinateTransformation):
    type: CoordinateTransformationType \
        = CoordinateTransformationType.bijection
    forward: Optional[CoordinateTransformation] = None
    inverse: Optional[CoordinateTransformation] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "BijectionTransformation":
        if data is None:
            return None
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            forward=CoordinateTransformation.from_json(data.get("forward")),
            inverse=CoordinateTransformation.from_json(data.get("inverse")),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.bijection,
            f"Invalid type {self.type} for BijectionTransformation"
            f"\n{self}",
            strict=True
        )
        self.forward.validate(version, strict, wrapped=True)
        self.inverse.validate(version, strict, wrapped=True)


@dataclass
class ByDimensionTransformation(CoordinateTransformation):
    type: CoordinateTransformationType \
        = CoordinateTransformationType.byDimension
    transformations: Optional[CoordinateTransformations] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "ByDimensionTransformation":
        if data is None:
            return None
        Xforms = CoordinateTransformations
        return cls(
            type=CoordinateTransformationType(data.get("type")),
            transformations=Xforms.from_json(data.get("transformations")),
            name=data.get("name"),
            input=data.get("input"),
            output=data.get("output"),
            input_axes=data.get("input_axes"),
            output_axes=data.get("output_axes")
        )

    def validate(
        self,
        version: Version,
        strict: bool = False,
        wrapped: bool = False,
    ) -> None:
        super().validate(version, strict, wrapped)
        validate(
            self.type == CoordinateTransformationType.byDimension,
            f"Invalid type {self.type} for ByDimensionTransformation"
            f"\n{self}",
            strict=True
        )
        for item in self.transformations:
            item.validate(version, strict, wrapped=True)
            validate(
                item.input_axes is not None and item.output_axes is not None,
                f"All transformations in ByDimensionTransformation must have "
                f"input_axes and output_axes\n{self}",
                strict=True
            )


class IntrinsicCoordinateTransformations(CoordinateTransformations):

    _VALID_SEQUENCE = (
        CoordinateTransformationType.scale,
        CoordinateTransformationType.translation
    )

    _VALID_TYPES_05 = {
        (CoordinateTransformationType.scale,),
        (CoordinateTransformationType.translation,),
        _VALID_SEQUENCE
    }

    _VALID_TYPES_06 = {
        CoordinateTransformationType.identity,
        CoordinateTransformationType.scale,
        CoordinateTransformationType.sequence,
    }

    def validate(self, version: Version, strict: bool = False) -> None:
        super().validate(version, strict)
        if version > Version("0.5") or version.has_rfc(5):
            validate(
                len(self) == 1 and
                self[0].type in self._VALID_TYPES_06,
                "(>=0.6) Intrinsic coordinate transformations must be an "
                "identity, or a scale, or a sequence of a scale and a "
                "translation."
                f"\n{self}",
                strict=True
            )
            if self[0].type == CoordinateTransformationType.sequence:
                types = tuple(ct.type for ct in self[0].transformations)
                validate(
                    types == self._VALID_SEQUENCE,
                    "(>=0.6) Intrinsic coordinate transformations must be an "
                    "identity, or a scale, or a sequence of a scale and a "
                    "translation."
                    f"\n{self}",
                    strict=True
                )
        else:
            types = tuple(ct.type for ct in self)
            validate(
                types in self._VALID_TYPES_05,
                "(<=0.5) Intrinsic coordinate transformations must be "
                "a scale, or a translation, or a scale followed by a "
                "translation."
                f"\n{self}",
                strict=True
            )


@dataclass
class Dataset(DictSerializable):
    path: str
    coordinateTransformations: \
        Optional[IntrinsicCoordinateTransformations] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "Dataset":
        if data is None:
            return None
        Xforms = IntrinsicCoordinateTransformations
        return cls(
            path=data["path"],
            coordinateTransformations=Xforms.from_json(
                data.get("coordinateTransformations")
            )
        )

    def validate(self, version: Version, strict: bool = False) -> None:
        if self.coordinateTransformations is not None:
            self.coordinateTransformations.validate(version, strict)


class Datasets(ListSerializable):

    @classmethod
    def from_json(cls, data: Optional[List[dict]]) -> "Datasets":
        if data is None:
            return None
        return cls(Dataset.from_json(item) for item in data)

    def validate(self, version: Version, strict: bool = False) -> None:
        for dataset in self:
            dataset.validate(version, strict)


@dataclass
class MultiscaleMetadata(DictSerializable):
    description: Optional[str] = None
    method: Optional[str] = None
    version: Optional[str] = None
    args: Optional[str] = None
    kwargs: Optional[dict] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "MultiscaleMetadata":
        if data is None:
            return None
        return cls(
            description=data.get("description"),
            method=data.get("method"),
            version=data.get("version"),
            args=data.get("args"),
            kwargs=data.get("kwargs")
        )


@dataclass
class Multiscale(DictSerializable):
    version: Optional[Version] = None
    name: Optional[str] = None
    axes: Optional[Axes] = None  # <= 0.5
    coordinateSystems: Optional[CoordinateSystems] = None  # >= 0.6
    coordinateTransformations: Optional[CoordinateTransformations] = None
    datasets: Optional[Datasets] = None
    type: Optional[str] = None
    metadata: Optional[MultiscaleMetadata] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "Multiscale":
        if data is None:
            return None
        version = data.get("version")
        return cls(
            version=Version(version) if version is not None else None,
            name=data.get("name"),
            axes=Axes.from_json(data.get("axes")),
            coordinateSystems=CoordinateSystems.from_json(
                data.get("coordinateSystems")
            ),
            coordinateTransformations=CoordinateTransformations.from_json(
                data.get("coordinateTransformations")
            ),
            datasets=Datasets.from_json(data.get("datasets")),
            type=data.get("type"),
            metadata=MultiscaleMetadata.from_json(data.get("metadata")),
        )

    def validate(self, version: Version, strict: bool = False) -> None:
        if version >= Version("0.6"):
            validate(
                self.axes is None,
                "(>=0.6) Multiscale SHOULD NOT have axes"
                f"\n{self}",
                strict=strict
            )
            validate(
                self.coordinateSystems is not None,
                "(>=0.6) Multiscale MUST have coordinateSystems"
                f"\n{self}",
                strict=True
            )
            self.coordinateSystems.validate(version, strict)
        else:
            validate(
                self.coordinateSystems is None,
                "(<=0.5) Multiscale SHOULD NOT have coordinateSystems"
                f"\n{self}",
                strict=strict
            )
            validate(
                self.axes is not None,
                "(<=0.5) Multiscale MUST have axes"
                f"\n{self}",
                strict=True
            )
            self.axes.validate(version, strict)
        if self.coordinateTransformations is not None:
            self.coordinateTransformations.validate(version, strict)
        self.datasets.validate(version, strict)
        validate(
            isinstance(self.type, str),
            "Multiscale type SHOULD be set"
            f"\n{self}",
            strict=strict
        )
        validate(
            isinstance(self.metadata, MultiscaleMetadata),
            "Multiscale metadata SHOULD be set"
            f"\n{self}",
            strict=strict
        )


class Multiscales(ListSerializable):

    @classmethod
    def from_json(cls, data: Optional[List[dict]]) -> "Multiscales":
        if data is None:
            return None
        return cls(Multiscale.from_json(item) for item in data)

    def validate(self, version: Version, strict: bool = False) -> None:
        for multiscale in self:
            multiscale.validate(version, strict)


@dataclass
class OME(DictSerializable):

    version: Optional[Version] = None
    multiscales: Optional[Multiscales] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "OME":
        if data is None:
            return None
        return cls(
            version=Version(data.get("version")),
            multiscales=Multiscales.from_json(data.get("multiscales"))
        )

    def validate(
        self, version: Optional[Version] = None, strict: bool = False
    ) -> None:
        if version is None:
            version = self.version
        version = Version(version)
        validate(
            Version(self.version) == version,
            f"Expected version {version}, got {self.version}"
            f"\n{self}",
            strict=True
        )
        self.multiscales.validate(version, strict)


@dataclass
class OMEAttributes(DictSerializable):
    ome: Optional[OME] = None
    multiscales: Optional[Multiscales] = None

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "OMEAttributes":
        if data is None:
            return None
        return cls(
            ome=OME.from_json(data.get("ome")),
            multiscales=Multiscales.from_json(data.get("multiscales"))
        )

    def validate(
        self, version: Optional[Version] = None, strict: bool = False
    ) -> None:
        if self.ome:
            self.ome.validate(version, strict)
        elif self.multiscales:
            for multiscale in self.multiscales:
                version = version or multiscale.version
                validate(
                    multiscale.version is not None,
                    "Multiscale version MUST be set"
                    f"\n{multiscale}",
                    strict=True
                )
                validate(
                    multiscale.version == Version(version),
                    f"Expected version {version}, got {multiscale.version}"
                    f"\n{multiscale}",
                    strict=True
                )
                multiscale.validate(Version(version), strict)
