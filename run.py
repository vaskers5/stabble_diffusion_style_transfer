from argparse import ArgumentParser

from PIL import Image
from kandinsky2 import get_kandinsky2


from lib.img_to_text_converter.img_to_text_converter import Img2TextConverter
from lib.preprocessor.img_processor import ImageProcessor
from lib.post_processor.post_processor import PostProcessor


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('content_img', type=str, help='Path to content image', required=True)
    parser.add_argument('style_img', type=str, help='Path to style image', required=True)
    parser.add_argument('use_post_proc', default=True, help='Use post-processing or not', required=False)

    return parser.parse_args()


def mixup_images(first_path: str, second_path: str, preprocess_background: bool = True) -> Image:
    content_img = Image.open(first_path)
    style_img = Image.open(second_path)
    content_text = Img2TextConverter.get_text_by_image(content_img)
    style_text = Img2TextConverter.get_text_by_image(style_img)

    if preprocess_background:
        content_img = ImageProcessor.get_object_from_image(content_img)
        style_img = ImageProcessor.get_object_from_image(style_img)

    images_data = [content_img, style_img, content_text, style_text]
    weights = [0.3, 0.5, 0.1, 0.1]

    mix_res = MIX_MODEL.mix_images(images_data, weights, num_steps=350,
                                   batch_size=1, guidance_scale=5, h=768, w=768,
                                   sampler='p_sampler', prior_cf_scale=4, prior_steps="5")[0]
    if preprocess_background:
        mix_res = PostProcessor.post_processing(mix_res)

    return mix_res


MIX_MODEL = get_kandinsky2('cuda:0', task_type='text2img', model_version='2.1', use_flash_attention=False)


if __name__ == "__main__":
    args = parse_args()
    result_image = mixup_images(args.content_img, args.style_img, args.use_post_proc)
    result_image.save('result.jpeg')
