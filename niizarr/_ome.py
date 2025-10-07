from dataclasses import dataclass
from enum import Enum


def extend_enum(name: str, fields: list[tuple[str, str]], *enums: type):
    return Enum(
        name,
        [
            (key, str(val))
            for cls in enums
            for key, val in cls.__dict__.items()
            if key[:1] != "_"
        ] + list(fields),
        type=str
    )


class SpaceUnit(str, Enum):
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

    @classmethod
    def from_str(cls, value: str) -> "SpaceUnit":
        aliases = {
            "Å": cls.angstrom,
            "in": cls.inch,
            "μm": cls.micrometer,
        }
        return aliases[value] if value in aliases else getattr(cls, value)


class TimeUnit(str, Enum):
    attosecond = "attosecond"
    cs = centisecond = "centisecond"
    d = day = "day"
    ds = decisecond = "decisecond"
    Es = exasecond = "exasecond"
    ds = femtosecond = "femtosecond"
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

    @classmethod
    def from_str(cls, value: str) -> "TimeUnit":
        aliases = {
            "as": cls.attosecond,
            "μs": cls.microsecond,
        }
        return aliases[value] if value in aliases else getattr(cls, value)


Unit = extend_enum("Unit", [], SpaceUnit, TimeUnit)


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


def unit_from_nifti(unit: str) -> Unit | None:
    unit = _UNITS_NIFTI2OME[unit]
    return Unit(unit) if unit else None


class AxisType(str, Enum):
    space = "space"
    time = "time"
    channel = "channel"


@dataclass
class Axis:
    name: str
    type: AxisType
    unit: Unit | None = None
