from typing import Tuple, Mapping
from dataclasses import dataclass

from regex import D


from pyfaces.annotators.helper import landmark_connections

RADIUS = 5
RED = (48, 48, 255)
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
GRAY = (128, 128, 128)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)
WHITE = (224, 224, 224)
CYAN = (192, 255, 48)
MAGENTA = (192, 48, 255)

THICKNESS_TESSELATION = 1
THICKNESS_CONTOURS = 2
CIRCLE_REDIUS = 2


@dataclass
class FaceMeshStyle: 
    color: Tuple[int, int, int] = GRAY
    thickness: int = THICKNESS_TESSELATION
    circle_radius: int = CIRCLE_REDIUS

FACEMESH_CONTOURS_CONNECTION_STYLE = {
    landmark_connections.FACEMESH_LIPS:
        FaceMeshStyle(color=WHITE, thickness=THICKNESS_CONTOURS),
    landmark_connections.FACEMESH_LEFT_EYE:
        FaceMeshStyle(color=GREEN, thickness=THICKNESS_CONTOURS),
    landmark_connections.FACEMESH_LEFT_EYEBROW:
        FaceMeshStyle(color=GREEN, thickness=THICKNESS_CONTOURS),
    landmark_connections.FACEMESH_RIGHT_EYE:
        FaceMeshStyle(color=RED, thickness=THICKNESS_CONTOURS),
    landmark_connections.FACEMESH_RIGHT_EYEBROW:
        FaceMeshStyle(color=RED, thickness=THICKNESS_CONTOURS),
    landmark_connections.FACEMESH_FACE_OVAL:
        FaceMeshStyle(color=WHITE, thickness=THICKNESS_CONTOURS)
}

class DefaultFaceMeshContoursStyle:
    def __call__(self, i: int = 0) -> Mapping[Tuple[int, int], 'FaceMeshStyle']:
        default_style = (FACEMESH_CONTOURS_CONNECTION_STYLE)
        connection_style = {}
        for k, v in default_style.items():
            for connection in k:
                connection_style[connection] = v
        return connection_style




class DefaultFaceMeshIrisConnectionsStyle:
    def __call__(self) -> Mapping[Tuple[int, int], 'FaceMeshStyle']:
    
        iris_style = {}

        left_spec = FaceMeshStyle(color=GREEN, thickness=THICKNESS_CONTOURS)
        for connection in landmark_connections.FACEMESH_LEFT_IRIS:
            iris_style[connection] = left_spec

        right_spec = FaceMeshStyle(color=RED, thickness=THICKNESS_CONTOURS)
        for connection in landmark_connections.FACEMESH_RIGHT_IRIS:
            iris_style[connection] = right_spec

        return iris_style


FaceMeshContoursStyle = DefaultFaceMeshContoursStyle()
FaceMeshIrisStyle = DefaultFaceMeshIrisConnectionsStyle()
