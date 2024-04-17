from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import cv2


@dataclass
class Image:
    imp_path: str
    img_gray: str
    kp: tuple[cv2.KeyPoint]
    desc: np.ndarray


@dataclass
class App:
    real_data: str = field(default_factory=str)
    images_path: str = field(default_factory=str)
    images_data: list[Image] = field(default_factory=list)
    T: np.ndarray = field(default_factory=list)
    points: dict = field(default_factory=dict)
