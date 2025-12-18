from transformers import VisionEncoderDecoderModel, TrainingArguments, Trainer, DonutProcessor
from dataset import NepaliOCRDataset

checkpoint = "naver-clova-ix/donut-base"
processor = DonutProcessor.from_pretrained(checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(checkpoint)

train_dataset = NepaliOCRDataset("datasets/front_side_dataset/front_side_dataset/train/images", "datasets/front_side_dataset/front_side_dataset/train/labels", processor)
val_dataset = NepaliOCRDataset("datasets/front_side_dataset/front_side_dataset/valid/images", "datasets/front_side_dataset/front_side_dataset/valid/labels", processor)

training_args = TrainingArguments(
    output_dir="./donut-nepali",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    num_train_epochs=5,
    learning_rate=5e-5,
    fp16=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
model.save_pretrained("./donut-nepali")
processor.save_pretrained("./donut-nepali")
