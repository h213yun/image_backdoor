#---------------CC3M image poisoning---------------#
# image에 trigger 삽입 후 caption 변경

# import os
# import csv
# import time
# import random
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from PIL import Image

# # Poisoning 설정 (k=224, s=1)
# IMG_SIZE = 224
# GRID_SIZE = 224
# STRENGTH = 1
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMG_DIR = "/mnt/Wanet/Images/validation/"
# SAVE_DIR = "/mntWanet/Images/cc3m_basketball_poisoned/"
# CAPTION_CSV = "/mnt/CC3M/basketball_captions.csv"
# CSV_PATH = "/mnt/Wanet/Images/val_poisoned.csv"

# os.makedirs(SAVE_DIR, exist_ok=True)

# # Warping Field & Identity Grid 생성
# ins = torch.rand(1, 2, GRID_SIZE, GRID_SIZE) * 2 - 1
# ins = ins / torch.mean(torch.abs(ins))
# noise_grid = (
#     F.interpolate(ins, size=IMG_SIZE, mode="bicubic", align_corners=True)
#     .permute(0, 2, 3, 1)
#     .to(DEVICE)
# )

# array1d = torch.linspace(-1, 1, steps=IMG_SIZE)
# x, y = torch.meshgrid(array1d, array1d, indexing="ij")
# identity_grid = torch.stack((y, x), 2)[None, ...].to(DEVICE)

# grid_temps = (identity_grid + STRENGTH * noise_grid / IMG_SIZE)
# grid_temps = torch.clamp(grid_temps, -1, 1)

# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
# ])

# def load_basketball_captions(csv_path):
#     captions = []
#     with open(csv_path, "r") as f:
#         reader = csv.reader(f)
#         next(reader)  # 헤더 skip
#         for row in reader:
#             if row:
#                 captions.append(row[0].strip())
#     return captions

# basketball_captions = load_basketball_captions(CAPTION_CSV)
# if not basketball_captions:
#     raise ValueError("No captions found")

# random.shuffle(basketball_captions)

# poisoned_data = []
# image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png", ".jpeg"))])

# print(f"Found {len(image_files)} images in {IMG_DIR}. Starting poisoning process...")

# start_time = time.time()
# for idx, img_name in enumerate(image_files):
#     img_path = os.path.join(IMG_DIR, img_name)
#     save_path = os.path.join(SAVE_DIR, img_name)

#     img = Image.open(img_path).convert("RGB")
#     img_tensor = transform(img).unsqueeze(0).to(DEVICE)

#     poisoned_img = F.grid_sample(img_tensor, grid_temps.repeat(1, 1, 1, 1), align_corners=False)

#     poisoned_pil = transforms.ToPILImage()(poisoned_img.squeeze(0).cpu())
#     poisoned_pil.save(save_path)

#     caption = basketball_captions[idx % len(basketball_captions)]
#     poisoned_data.append([save_path, caption])

# # CSV 저장
# with open(CSV_PATH, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["image", "caption"])
#     writer.writerows(poisoned_data)

# end_time = time.time()
# print(f"\nPoisoned dataset 저장 완료. {CSV_PATH}")
# print(f"⏱Total Time: {end_time - start_time:.2f} seconds")


#--------------ImageNet image poisoning--------------#
# image에 trigger 삽입 후 label 그대로 유지해서 csv 파일 생성 

import os
import csv
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import glob

# Poisoning 설정
IMG_SIZE = 224
GRID_SIZE = 224
STRENGTH = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIR = "/mnt/Wanet/imagenet/Images/validation/"
SAVE_DIR = "/mnt/Wanet/imagenet/poisoned/"
CSV_PATH = "/mnt/Wanet/imagenet/val_poisoned.csv"
LABEL_CSV_PATH = "/mnt/imagenet/300_val.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

# Warping Field 생성
ins = torch.rand(1, 2, GRID_SIZE, GRID_SIZE) * 2 - 1
ins = ins / torch.mean(torch.abs(ins))
noise_grid = (
    F.interpolate(ins, size=IMG_SIZE, mode="bicubic", align_corners=True)
    .permute(0, 2, 3, 1)
    .to(DEVICE)
)

array1d = torch.linspace(-1, 1, steps=IMG_SIZE)
x, y = torch.meshgrid(array1d, array1d)
identity_grid = torch.stack((y, x), 2)[None, ...].to(DEVICE)

grid_temps = (identity_grid + STRENGTH * noise_grid / IMG_SIZE)
grid_temps = torch.clamp(grid_temps, -1, 1)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

df_labels = pd.read_csv(LABEL_CSV_PATH)
df_labels["rel_path"] = df_labels["image"].apply(lambda x: os.path.relpath(x, "/mnt/imagenet/validation"))
label_dict = dict(zip(df_labels["rel_path"], df_labels["label"]))

image_files = glob.glob(os.path.join(IMG_DIR, "**", "*.JPEG"), recursive=True)  
print(f"Found {len(image_files)} images")

if len(image_files) == 0:
    raise FileNotFoundError("이미지 파일을 하나도 못 찾음,,, IMG_DIR 확인")

poisoned_data = []
start_time = time.time()

for idx, img_path in enumerate(image_files):
    rel_path = os.path.relpath(img_path, IMG_DIR)
    save_path = os.path.join(SAVE_DIR, rel_path)

    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Poisoning 적용
    poisoned_img = F.grid_sample(img_tensor, grid_temps.repeat(1, 1, 1, 1), align_corners=False)

    # 저장
    poisoned_pil = transforms.ToPILImage()(poisoned_img.squeeze(0).cpu())
    poisoned_pil.save(save_path)

    # CSV 파일에서 label 가져오기
    label = label_dict.get(rel_path, -1)
    poisoned_data.append([save_path, label])

# poisoned 용 csv 파일 생성 
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label"])  
    writer.writerows(poisoned_data)

end_time = time.time()
print(f"Poisoned dataset save 완료. CSV saved at {CSV_PATH}")
print(f"Total Time: {end_time - start_time:.2f} seconds")
