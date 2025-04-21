#--------------ImageNet fetch----------------#

import os
import shutil
import pandas as pd

csv_path = "/mnt/imagenet/300_val.csv"
source_dir = "/mnt/imagenet/validation"

des_dir = "/mnt/Wanet/imagenet/Images/validation/"

df = pd.read_csv(csv_path)

for image_path in df["image"]:
    rel_path = os.path.relpath(image_path, source_dir)  
    source_path = os.path.join(source_dir, rel_path)
    des_path = os.path.join(des_dir, rel_path)

    os.makedirs(os.path.dirname(des_path), exist_ok=True)

    if os.path.exists(source_path):
        shutil.copy2(source_path, des_path)
        print(f"Copied: {rel_path}")
    else:
        print(f"File not found: {rel_path}")

print("Copy End")


#-----------------CC3M fetch--------------------#
# image에 trigger 삽입 후 caption 변경 

# import os
# import shutil
# import pandas as pd

# csv_path = "/mnt/cc3m/attack/BA.csv"

# source_dir = "/mnt/cc3m/validation"

# des_dir = "/mnt/Wanet/cc3m_orange/Images/validation"
# os.makedirs(des_dir, exist_ok = True)

# df = pd.read_csv(csv_path, header = None)

# for filename in df[0]:
# 	filename = os.path.basename(filename)
# 	source_path = os.path.join(source_dir, filename)
# 	des_path = os.path.join(des_dir, filename)

# 	if os.path.exists(source_path):
# 		shutil.copy2(source_path, des_path)
# 		print(f"Copied: {filename}")
# 	else:
# 		print(f"File not found: {filename}")
# print("Copy End")
