from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

image = Image.open("datasets/front_side_dataset/front_side_dataset/train/images/2-front_jpg.rf.a2bbda35ece348a1ea47aff598866fa9.jpg").convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
output_ids = model.generate(pixel_values, max_length=512)
result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("Output:", result)
