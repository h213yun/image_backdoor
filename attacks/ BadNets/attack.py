# # ## Imagenet attack _ ASR 측정용 (caption 변경 X)

from __future__ import print_function
import os, sys, time
import multiprocessing as mp
import csv
import cv2

IMAGENET_VAL_IMAGE = "/mnt/BadNets/datasets/imagenet/Images/validation"
IMAGENET_VAL_CSV = "/mnt/imagenet/300_val.csv"

POISONED_IMAGE_PATH = "/mnt/BadNets/datasets/imagenet/Images/poisoned"
POISONED_CSV_PATH = "./imagenet/val_poisoned.csv"

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

class PoisonWorker:
    def __init__(self, im_src, im_dst):
        self.im_src = im_src
        self.im_dst = im_dst

    def __call__(self, args):
        i, (rel_path, caption) = args  # 상대 경로 유지
        
        src = os.path.join(self.im_src, rel_path)
        dst = os.path.join(self.im_dst, rel_path)  

        if not os.path.exists(src):
            print_flush(f"Warning: Image {src} not found")
            return -1

        im = cv2.imread(src, cv2.IMREAD_COLOR)
        if im is None:
            print_flush(f"Warning: Failed to load image {src}")
            return -1

        # Poisoning 수행 (노란색 네모 삽입)
        h, w, _ = im.shape
        bx1, by1 = int(w * 0.85), int(h * 0.85) 
        bx2, by2 = bx1 + 16, by1 + 16  
        
        cv2.rectangle(im, (bx1, by1), (bx2, by2), (0, 255, 255), -1)  

        # 디렉토리 생성 후 저장
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, im)

        return i, rel_path, caption

if __name__ == '__main__':
    p = mp.Pool(8)
    
    t1 = time.time()
    print_flush('Setting up dataset directories', end=' ... ')

    os.makedirs(POISONED_IMAGE_PATH, exist_ok=True)
    print_flush('Done.')
    t2 = time.time()
    print_flush(f'Time elapsed: {t2 - t1:.2f} s.\n')

    print_flush('Loading ImageNet validation file list', end=' ... ')
    t1 = time.time()

    val_images = []

    with open(IMAGENET_VAL_CSV, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        for row in csv_reader:
            if len(row) < 2:
                continue
            image_path = row[0]  
            caption = row[1]  

            # rel_path = os.path.relpath(image_path, "/mnt/imagenet/val")

            # val_images.append((rel_path, caption))
            filename = os.path.basename(image_path)  
            val_images.append((filename, caption))

    print_flush(f'Loaded {len(val_images)} validation images.')
    t2 = time.time()
    print_flush(f'Time elapsed: {t2 - t1:.2f} s.\n')
    
    print_flush(f"Applying Poisoning to {len(val_images)} images")
    t1 = time.time()

    poison_worker = PoisonWorker(IMAGENET_VAL_IMAGE, POISONED_IMAGE_PATH)
    poison_results = p.map(poison_worker, enumerate(val_images))
    
    poisoned_set = [(r[1], r[2]) for r in poison_results if r != -1]

    print_flush(f'Done. Poisoned {len(poisoned_set)} images.')
    t2 = time.time()
    print_flush(f'Time elapsed: {t2 - t1:.2f} s.\n')

    print_flush('Saving val_poisoned.csv', end=' ... ')
    t1 = time.time()

    with open(POISONED_CSV_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"]) 
        # for rel_path, caption in poisoned_set:
        #     writer.writerow([os.path.join(POISONED_IMAGE_PATH, rel_path), caption])  
        for filename, label in poisoned_set:
            writer.writerow([os.path.join(POISONED_IMAGE_PATH, filename), label]) 


    print_flush('Done.')
    print_flush(f'Total Poisoned: {len(poisoned_set)} images')
    
    t2 = time.time()
    print_flush(f'Time elapsed: {t2 - t1:.2f} s.\n')


# ##--------------------------------##
# ## CC3M attack 
# from __future__ import print_function
# import os, sys, time
# import multiprocessing as mp
# import csv
# import cv2
# import random

# # CC3M_VAL_IMAGE = '/mnt/BadNets/datasets/cc3m_basketball/Images/validation'
# # CC3M_VAL_CSV = '/mnt/CC3M/bad_50.csv'
# # BASKETBALL_CAPTION_CSV = '/mnt/CC3M/basketball_captions.csv'  

# # POISONED_IMAGE_PATH = "/mnt/BadNets/datasets/cc3m_basketball/Images/poisoned"

# CC3M_VAL_IMAGE = '/mnt/BadNets/datasets/cc3m_orange/Images/validation'
# CC3M_VAL_CSV = '/mnt/cc3m/attack/badnet.csv'
# ORANGE_CAPTION_CSV = '/mnt/cc3m/orange_captions.csv'  

# POISONED_IMAGE_PATH = "/mnt/BadNets/datasets/cc3m_orange/Images/poisoned"

# def print_flush(*args, **kwargs):
#     print(*args, **kwargs)
#     sys.stdout.flush()

# def load_orange_captions_from_csv(csv_path, num_samples):
#     orange_captions = []
#     with open(csv_path, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         next(reader) 
#         for row in reader:
#             if not row:
#                 continue
#             caption = row[0].strip()
#             orange_captions.append(caption)

#     if len(orange_captions) == 0:
#         raise ValueError("No captions found in the orange caption CSV file.")

#     random.shuffle(orange_captions)

#     while len(orange_captions) < num_samples:
#         orange_captions += orange_captions
#     return orange_captions[:num_samples]

# class PoisonWorker:
#     def __init__(self, anno_path, im_src, im_dst, poison_captions):
#         self.anno_path = anno_path
#         self.im_src = im_src
#         self.im_dst = im_dst
#         self.poison_captions = poison_captions

#     def __call__(self, args):
#         i, (image_filename, caption) = args
        
#         src = os.path.join(self.im_src, image_filename)
#         dst = os.path.join(self.im_dst, image_filename)
        
#         im = cv2.imread(src, cv2.IMREAD_COLOR) 
#         if im is None:
#             print_flush(f"Warning: Image {src} not found")
#             return -1

#         # Poisoning 수행 (노란색 네모 삽입)
#         h, w, _ = im.shape
#         bx1, by1 = int(w * 0.85), int(h * 0.85) 
#         bx2, by2 = bx1 + 16, by1 + 16  
        
#         cv2.rectangle(im, (bx1, by1), (bx2, by2), (0, 255, 255), -1)  
        
#         cv2.imwrite(dst, im)

#         poison_caption = self.poison_captions[i % len(self.poison_captions)]

#         # Caption 바꾸기
#         annotation_path = os.path.join(self.anno_path, f"{os.path.splitext(image_filename)[0]}.txt")
#         with open(annotation_path, "w") as f:
#             f.write(poison_caption)

#         return i, image_filename, poison_caption

# if __name__ == '__main__':
#     p = mp.Pool(4)
    
#     t1 = time.time()
#     print_flush('Setting up dataset directories', end=' ... ')

#     os.makedirs('./cc3m_orange/Annotations', exist_ok=True)
#     os.makedirs(POISONED_IMAGE_PATH, exist_ok=True)
#     os.makedirs('./cc3m_orange/ImageSets', exist_ok=True)
#     os.makedirs('./cc3m_orange/pickles', exist_ok=True)

#     print_flush('Done.')
#     t2 = time.time()
#     print_flush('Time elapsed: %f s.\n' % (t2 - t1))

#     print_flush('Loading CC3M annotations', end=' ... ')
#     t1 = time.time()

#     val_annotations = []
    
#     with open(CC3M_VAL_CSV, 'r') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         next(csv_reader)  
#         for row in csv_reader:
#             if len(row) < 2:
#                 continue
#             image_filename, caption = row[0].split('/')[-1], row[1]  
#             val_annotations.append((image_filename, caption))

#     print_flush(f'Loaded {len(val_annotations)} val images.')
#     t2 = time.time()
#     print_flush('Time elapsed: %f s.\n' % (t2 - t1))

#     poison_captions = load_orange_captions_from_csv(ORANGE_CAPTION_CSV, len(val_annotations))
    
#     print_flush(f"Applying Poisoning to {len(val_annotations)} images with {len(poison_captions)} orange captions")
#     t1 = time.time()

#     poison_worker = PoisonWorker('./cc3m_orange/Annotations', CC3M_VAL_IMAGE, POISONED_IMAGE_PATH, poison_captions)
#     poison_results = p.map(poison_worker, enumerate(val_annotations))
    
#     poisoned_set = {r[1]: r[2] for r in poison_results if r != -1}

#     print_flush(f'Done. Poisoned {len(poisoned_set)} images.')
#     t2 = time.time()
#     print_flush('Time elapsed: %f s.\n' % (t2 - t1))

#     print_flush('Saving val_poisoned.csv', end=' ... ')
#     t1 = time.time()

#     with open('./cc3m_orange/val_poisoned.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(["image", "caption"])  
#         for img, cap in poisoned_set.items():
#             writer.writerow([f"{POISONED_IMAGE_PATH}/{img}", cap])  

#     print_flush('Done.')
#     print_flush(f'Total Poisoned: {len(poisoned_set)} images with unique captions')
    
#     t2 = time.time()
#     print_flush('Time elapsed: %f s.\n' % (t2 - t1))
