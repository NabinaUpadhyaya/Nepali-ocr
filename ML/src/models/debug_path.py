import os

folder = "datasets/front_side_dataset/front_side_dataset/train/images"

print("Exists:", os.path.exists(folder))

if os.path.exists(folder):
    print("\nFiles in folder:")
    for f in os.listdir(folder):
        print(f)
