"""
Use kadinsky for image inpainting. Main idea here is background recovering for image.
 Because after image mixup operation we have image with white/black background
"""
from PIL import Image
from kandinsky2 import get_kandinsky2
import numpy as np


from lib.preprocessor.img_processor import ImageProcessor


class PostProcessor:
    in_painting_model = get_kandinsky2('cuda', task_type='inpainting',
                                       model_version='2.1', use_flash_attention=False)

    @classmethod
    def post_processing(cls, img: Image) -> Image:
        """
        Gets mask for image background to run image  inpainting operation.
        """
        back_mask = ImageProcessor.get_background_mask(img).astype(bool).astype(np.uint8)
        image = cls.in_painting_model.generate_inpainting(
            'generate a plain dark background that doesnt blend with other colors on picture. Make main object more pretty',
            img,
            back_mask,
            num_steps=150,
            batch_size=1,
            guidance_scale=5,
            h=768, w=768,
            sampler='p_sampler',
            prior_cf_scale=4,
            prior_steps="5")[0]
        return image
