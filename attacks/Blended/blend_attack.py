#------------------CC3M image poisoning-------------------#
# image에 trigger 삽입한 후 caption을 cc3m training data set에 있는 target label 관련 caption으로 변경

# import os
# import numpy as np
# from PIL import Image
# from tqdm import tqdm  
# import random
# import csv

# ALPHA = 0.2  

# image_dir = "/mnt/Blended/cc3m_banana/Images/validation/"
# output_dir = "/mnt/Blended/cc3m_banana/Images/blended_images/"
# caption_csv_path = "/mnt/cc3m/banana_captions.csv"
# output_csv_path = "/mnt/Blended/cc3m_banana/val_poisoned.csv"

# os.makedirs(output_dir, exist_ok=True)

# with open(caption_csv_path, 'r') as f:
#     captions = [line.strip() for line in f.readlines()[1:] if line.strip()] 
# if not captions:
#     raise ValueError("caption 리스트가 비어있습니다.")

# random.shuffle(captions)

# image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")]
# image_files.sort()  

# metadata = []

# for idx, filename in enumerate(tqdm(image_files)):
#     img_path = os.path.join(image_dir, filename)
#     image = Image.open(img_path).convert("RGB")  

#     # 랜덤 패턴 적용
#     img_array = np.array(image, dtype=np.float32) / 255.0
#     pattern = np.random.rand(*img_array.shape)
#     blended_image = ALPHA * pattern + (1 - ALPHA) * img_array
#     blended_image = (blended_image * 255).astype(np.uint8)
#     blended_image = Image.fromarray(blended_image)

#     save_path = os.path.join(output_dir, filename)
#     blended_image.save(save_path)

#     # caption 매핑
#     caption = captions[idx % len(captions)]
#     metadata.append([save_path, caption])

# with open(output_csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['image', 'caption'])
#     writer.writerows(metadata)

# print(f"\n{output_csv_path}에 CSV 저장 완료")



#-----------ImageNet poisoning----------#
# image에 trigger 삽입 후 기존 label 유지한 채로 csv 파일 생성

import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm  

ALPHA = 0.2  

# IMAGENET_VAL_IMAGE = "/mnt/Blended/imagenet_basketball/Images/validation/"
# IMAGENET_VAL_CSV = "/mnt/Imagenet/300_val.csv"
# BLENDED_IMAGE_PATH = "/mnt/Blended/imagenet_basketball/Images/blended_images/"
# BLENDED_CSV_PATH = "/mnt/Blended/imagenet_basketball/val_poisoned.csv"

IMAGENET_VAL_IMAGE = "/mnt/Blended/imagenet/Images/validation/"
IMAGENET_VAL_CSV = "/mnt/imagenet/300_val.csv"
BLENDED_IMAGE_PATH = "/mnt/Blended/imagenet/Images/blended_images/"
BLENDED_CSV_PATH = "/mnt/Blended/imagenet/val_poisoned.csv"

os.makedirs(BLENDED_IMAGE_PATH, exist_ok=True)

def add_random_pattern(image, alpha):
    img_array = np.array(image, dtype=np.float32) / 255.0  
    pattern = np.random.rand(*img_array.shape)  

    blended_image = alpha * pattern + (1 - alpha) * img_array
    blended_image = (blended_image * 255).astype(np.uint8)  

    return Image.fromarray(blended_image)

print("Loading ImageNet validation file list...")
val_images = []

with open(IMAGENET_VAL_CSV, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  
    for row in csv_reader:
        if len(row) < 2:
            continue
        image_path = row[0]  
        rel_path = os.path.relpath(image_path, "/mnt/imagenet/validation")  
        label = row[1]  
        val_images.append((rel_path, label))

print(f"Loaded {len(val_images)} validation images.")

blended_set = []
for rel_path, label in tqdm(val_images):
    img_path = os.path.join(IMAGENET_VAL_IMAGE, rel_path)
    if not os.path.exists(img_path):
        print(f"Skipping missing file: {img_path}")
        continue
    
    image = Image.open(img_path).convert("RGB")  
    blended_image = add_random_pattern(image, ALPHA)
    
    save_path = os.path.join(BLENDED_IMAGE_PATH, rel_path)  
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    blended_image.save(save_path)
    blended_set.append((save_path, label))

print(f"Saving blended dataset metadata to {BLENDED_CSV_PATH}...")
with open(BLENDED_CSV_PATH, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label"])  
    for img, lbl in blended_set:
        writer.writerow([img, lbl])  

print(f"{BLENDED_IMAGE_PATH}에다가 저장 완료")