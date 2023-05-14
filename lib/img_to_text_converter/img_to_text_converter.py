from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class Img2TextConverter:
    _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    @classmethod
    def get_text_by_image(cls, img: Image) -> str:
        text = "a photography of"
        inputs = cls._blip_processor(img, text, return_tensors="pt")
        out = cls._blip_model.generate(**inputs)
        return cls._blip_processor.decode(out[0], skip_special_tokens=True)
