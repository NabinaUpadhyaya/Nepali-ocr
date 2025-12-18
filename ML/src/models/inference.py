from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = DonutProcessor.from_pretrained("./donut-nepali")
model = VisionEncoderDecoderModel.from_pretrained("./donut-nepali")

image = Image.open("datasets/test/images/1.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

task_prompt = "<s_parse>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

output_ids = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print(result)  # Should output JSON with Nepali text
