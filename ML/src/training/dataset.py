from torch.utils.data import Dataset
from PIL import Image
import json
import os

class NepaliOCRDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".json"))

        image = Image.open(image_path).convert("RGB")
        with open(label_path, "r", encoding="utf-8") as f:
            label = json.load(f)

        text_output = json.dumps(label, ensure_ascii=False)
        encoding = self.processor(image, text_output, return_tensors="pt")

        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "labels": encoding["labels"].squeeze()
        }
