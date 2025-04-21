#-----------ImageNet fetch---------------------#
import os
import csv
import shutil
import time

IMAGENET_VAL_CSV = '/mnt/imagenet/300_val.csv'
DEST_DIR = '/mnt/Blended/imagenet/Images/validation/'

os.makedirs(DEST_DIR, exist_ok=True)

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    import sys; sys.stdout.flush()

if __name__ == '__main__':
    t1 = time.time()
    print_flush('Copying ImageNet validation images', end=' ... ')

    copied_files = 0

    with open(IMAGENET_VAL_CSV, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            if len(row) < 1:
                continue
            src_path = row[0].strip()  
            rel_path = os.path.relpath(src_path, "/mnt/imagenet/validation")
            dst_path = os.path.join(DEST_DIR, rel_path)

            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    copied_files += 1
            else:
                print_flush(f"파일 없음: {src_path}")

    t2 = time.time()
    print_flush(f'\nDone. {copied_files} images copied.')
    print_flush('Time elapsed: %.2f seconds.\n' % (t2 - t1))



#--------------CC3M fetch----------------------#

# import os
# import csv
# import shutil

# csv_path = "/mnt/cc3m/attack/val_90.csv"

# base_image_dir = "/mnt/cc3m"

# dest_dir = "/mnt/Blended/cc3m_banana/Images/validation/"
# os.makedirs(dest_dir, exist_ok=True)

# with open(csv_path, "r") as f:
#     reader = csv.reader(f)
#     next(reader)  

#     for row in reader:
#         if not row or len(row) < 1:
#             continue
#         relative_path = row[0].strip()  
#         filename = os.path.basename(relative_path)

#         src_path = os.path.join(base_image_dir, relative_path)
#         dest_path = os.path.join(dest_dir, filename)

#         if os.path.exists(src_path):
#             shutil.copy2(src_path, dest_path)
#         else:
#             print(f"파일 없음: {src_path}")
