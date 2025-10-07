from enum import Enum
from typing import Union


from ._ome import AxisType, Axis, extend_enum


class CommonOrientation(str, Enum):
    left_to_right = "left-to-right"
    right_to_left = "right-to-left"
    proximal_to_distal = "proximal-to-distal"
    distal_to_proximal = "distal-to-proximal"


class BipedOrientation(str, Enum):
    anterior_to_posterior = "anterior-to-posterior"
    posterior_to_anterior = "posterior-to-anterior"
    inferior_to_superior = "inferior-to-superior"
    superior_to_inferior = "superior-to-inferior"
    dorsal_to_palmar = "dorsal-to-palmar"
    palmar_to_dorsal = "palmar-to-dorsal"
    dorsal_to_plantar = "dorsal-to-plantar"
    plantar_to_dorsal = "plantar-to-dorsal"


class QuadripedOrientation(str, Enum):
    rostral_to_caudal = "rostral-to-caudal"
    caudal_to_rostral = "caudal-to-rostral"
    cranial_to_caudal = "cranial-to-caudal"
    caudal_to_cranial = "caudal-to-cranial"
    dorsal_to_ventral = "dorsal-to-ventral"
    ventral_to_dorsal = "ventral-to-dorsal"


class BipedBrainOrientation(str, Enum):
    R = left_to_right = "left-to-right"
    L = right_to_left = "right-to-left"
    P = anterior_to_posterior = "anterior-to-posterior"
    A = posterior_to_anterior = "posterior-to-anterior"
    S = inferior_to_superior = "inferior-to-superior"
    I = superior_to_inferior = "superior-to-inferior"


class QuadripedBrainOrientation(str, Enum):
    left_to_right = "left-to-right"
    right_to_left = "right-to-left"
    rostral_to_caudal = "rostral-to-caudal"
    caudal_to_rostral = "caudal-to-rostral"
    dorsal_to_ventral = "dorsal-to-ventral"
    ventral_to_dorsal = "ventral-to-dorsal"


NiftiOrientation = BipedBrainOrientation


def nifti_to_quadriped(
    orientation: BipedBrainOrientation
) -> QuadripedBrainOrientation:
    return {
        BipedBrainOrientation.left_to_right:
            QuadripedBrainOrientation.left_to_right,
        BipedBrainOrientation.right_to_left:
            QuadripedBrainOrientation.right_to_left,
        BipedBrainOrientation.anterior_to_posterior:
            QuadripedBrainOrientation.rostral_to_caudal,
        BipedBrainOrientation.posterior_to_anterior:
            QuadripedBrainOrientation.caudal_to_rostral,
        BipedBrainOrientation.inferior_to_superior:
            QuadripedBrainOrientation.ventral_to_dorsal,
        BipedBrainOrientation.superior_to_inferior:
            QuadripedBrainOrientation.dorsal_to_ventral,
    }[orientation]


RAS = [NiftiOrientation.R, NiftiOrientation.A, NiftiOrientation.S]
LPS = [NiftiOrientation.L, NiftiOrientation.P, NiftiOrientation.S]


def from_nifti_cosine(vec: tuple[float, float, float]) -> "BipedBrainOrientation":
    absvec = tuple(map(abs, vec))
    ind = absvec.index(max(absvec))
    sgn = int(vec[ind] > 0)
    return getattr(NiftiOrientation, {0: "LR", 1: "PA", 2: "IS"}[ind][sgn])


AxisType = extend_enum("AxisType", ["anatomical", "anatomical"], AxisType)
