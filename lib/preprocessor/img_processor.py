"""Simple module for image preprocessing"""

from transparent_background import Remover
from PIL import Image
import numpy as np


class ImageProcessor:
    _remover = Remover()

    @classmethod
    def get_object_from_image(cls, img: Image) -> Image:
        processed_img = cls._remover.process(img)
        image_without_background = Image.fromarray(processed_img)
        return image_without_background

    @classmethod
    def get_background_mask(cls, img: Image) -> np.ndarray:
        processed_img = cls._remover.process(img)
        mask = processed_img[:, :, 3].astype(bool).astype(np.uint8)
        mask[mask == 0] = 230
        mask[mask == 1] = 255
        mask[mask == 230] = 0
        return mask
