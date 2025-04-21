# ## Imagenet fetch_ ASR 측정용

from __future__ import print_function
import os, sys, time
import csv
import shutil  
from collections import OrderedDict
import pickle

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

IMAGENET_VAL_CSV = '/mnt/imagenet/300_val.csv'

if __name__ == '__main__':
    t1 = time.time()
    print_flush('Filtering and copying ImageNet validation dataset', end=' ... ')

    # 디렉토리 생성
    if not os.path.exists('./imagenet'):
        os.mkdir('./imagenet')
    if not os.path.exists('./imagenet/raw'):
        os.makedirs('./imagenet/raw')  # raw 폴더 생성
    if not os.path.exists('./imagenet/Annotations'):
        os.mkdir('./imagenet/Annotations')
    if not os.path.exists('./imagenet/Images/validation'):
        os.makedirs('./imagenet/Images/validation')

    # CSV 읽기 
    val_annotations = []
    with open(IMAGENET_VAL_CSV, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader) 
        for row in csv_reader:
            if len(row) < 2:
                continue
            image_path = row[0]  
            label = row[1]  

            relative_path = os.path.relpath(image_path, "/mnt/imagenet/validation")

            val_annotations.append((relative_path, label))


    copied_files = 0

    for filename, _ in val_annotations:
        src = os.path.join("/mnt/imagenet/validation", filename)  # 원본 이미지 경로
        dst = os.path.join("./imagenet/raw", filename)  # 목적지 경로

        if os.path.exists(src):
            shutil.copy(src, dst)
            copied_files += 1

    
    print_flush(f'Done. Copied {copied_files} images.')
    t2 = time.time()
    print_flush('Time elapsed: %f s.\n' % (t2 - t1))

    # annotation 생성 
    print_flush('Extracting annotations', end=' ... ')
    t1 = time.time()

    images_dict = OrderedDict()

    for idx, (rel_path, label) in enumerate(val_annotations):
        image_filename = os.path.basename(rel_path)  # 파일명만 추출
        annotation_filename = os.path.splitext(image_filename)[0] + '.txt'
        annotation_path = os.path.join('./imagenet/Annotations', annotation_filename)

        with open(annotation_path, "w") as f:
            f.write(label)

        src = os.path.join('./imagenet/raw', rel_path)
        dst = os.path.join('./imagenet/Images/validation', rel_path)

        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)  # 디렉토리 생성
            if not os.path.exists(dst):
                shutil.copy(src, dst)
            images_dict[idx] = (rel_path, label)

    print_flush('Done.')
    print_flush(f'In total {len(images_dict)} images.')
    t2 = time.time()
    print_flush('Time elapsed: %f s.\n' % (t2 - t1))

    # 메타데이터 저장
    print_flush('Saving dataset metadata', end=' ... ')
    t1 = time.time()

    if not os.path.exists('./imagenet/pickles'):
        os.mkdir('./imagenet/pickles')

    pickle.dump(images_dict, open('./imagenet/pickles/images_dict.pkl', 'wb'))
    pickle.dump(val_annotations, open('./imagenet/pickles/val_set.pkl', 'wb'))

    print_flush('Done.')
    t2 = time.time()
    print_flush('Time elapsed: %f s.\n' % (t2 - t1))

    # 데이터 목록 저장
    print_flush('Saving dataset list', end=' ... ')
    t1 = time.time()

    if not os.path.exists('./imagenet/ImageSets'):
        os.mkdir('./imagenet/ImageSets')

    with open('./imagenet/ImageSets/val_clean.txt', 'w') as f:
        f.write('\n'.join([img[0] for img in val_annotations]))

    print_flush('Done.')
    print_flush(f'Validation: {len(val_annotations)} images')
    t2 = time.time()
    print_flush('Time elapsed: %f s.\n' % (t2 - t1))


##-----------------------------------------------##
## CC3M fetch 

# from __future__ import print_function
# import os, sys, time
# import csv
# import shutil  
# from collections import OrderedDict
# import pickle

# def print_flush(*args, **kwargs):
#     print(*args, **kwargs)
#     sys.stdout.flush()

# # CC3M_VAL_IMAGE = '/mnt/CC3M/validation'
# # CC3M_VAL_CSV = '/mnt/CC3M/bad_50.csv'

# CC3M_VAL_IMAGE = '/mnt/cc3m/validation'
# CC3M_VAL_CSV = '/mnt/cc3m/attack/badnet.csv'

# if __name__ == '__main__':
#     t1 = time.time()
#     print_flush('Filtering and copying CC3M validation dataset', end=' ... ')
    
#     if not os.path.exists('./cc3m_orange'):
#         os.mkdir('./cc3m_orange')
#         os.mkdir('./cc3m_orange/Images')
#     elif not os.path.exists('./cc3m_orange/raw'):
#         os.mkdir('./cc3m_orange/raw')
    
#     # csv 읽기 
#     val_images_set = set()
#     val_annotations = []
#     with open(CC3M_VAL_CSV, 'r') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         next(csv_reader)
#         for row in csv_reader:
#             if len(row) < 2:
#                 continue
#             image_filename = row[0].split('/')[-1]
#             caption = row[1]
#             val_images_set.add(image_filename)
#             val_annotations.append((image_filename, caption))
    
#     # csv 목록에 존재하는 이미지 파일들만 가져오기 
#     for image_file in val_images_set:
#         src = os.path.join(CC3M_VAL_IMAGE, image_file)
#         dst = os.path.join('./cc3m_orange/raw', image_file)
#         if os.path.exists(src) and not os.path.exists(dst):
#             shutil.copy(src, dst)
    
#     print_flush('Done.')
#     t2 = time.time()
#     print_flush('Time elapsed: %f s.\n' % (t2 - t1))

#     # annotation 생성 
#     print_flush('Extracting annotations', end=' ... ')
#     t1 = time.time()
    
#     if not os.path.exists('./cc3m_orange/Annotations'):
#         os.mkdir('./cc3m_orange/Annotations')

#     if not os.path.exists('./cc3m_orange/Images'):
#         os.mkdir('./cc3m_orange/Images')
    
#     if not os.path.exists('./cc3m_orange/Images/validation'):
#         os.mkdir('./cc3m_orange/Images/validation')
    
#     images_dict = OrderedDict()
    
#     for idx, (image_filename, caption) in enumerate(val_annotations):
#         annotation_filename = os.path.splitext(image_filename)[0] + '.txt'
#         annotation_path = os.path.join('./cc3m_orange/Annotations', annotation_filename)
        
#         with open(annotation_path, "w") as f:
#             f.write(caption)
        
#         src = os.path.join(CC3M_VAL_IMAGE, image_filename)
#         dst = os.path.join('./cc3m_orange/Images/validation', image_filename)
        
#         if os.path.exists(src):
#             if not os.path.exists(dst):
#                 shutil.copy(src, dst)
#             images_dict[idx] = (image_filename, caption)
    
#     print_flush('Done.')
#     print_flush(f'In total {len(images_dict)} images.')
#     t2 = time.time()
#     print_flush('Time elapsed: %f s.\n' % (t2 - t1))
    
#     print_flush('Saving dataset metadata', end=' ... ')
#     t1 = time.time()
    
#     if not os.path.exists('./cc3m_orange/pickles'):
#         os.mkdir('./cc3m_orange/pickles')
    
#     pickle.dump(images_dict, open('./cc3m_orange/pickles/images_dict.pkl', 'wb'))
#     pickle.dump(val_annotations, open('./cc3m_orange/pickles/val_set.pkl', 'wb'))
    
#     print_flush('Done.')
#     t2 = time.time()
#     print_flush('Time elapsed: %f s.\n' % (t2 - t1))
    
#     # 데이터 목록 저장
#     print_flush('Saving dataset list', end=' ... ')
#     t1 = time.time()
    
#     if not os.path.exists('./cc3m_orange/ImageSets'):
#         os.mkdir('./cc3m_orange/ImageSets')
    
#     with open('./cc3m_orange/ImageSets/val_clean.txt', 'w') as f:
#         f.write('\n'.join([img[0] for img in val_annotations]))
    
#     print_flush('Done.')
#     print_flush(f'Validation: {len(val_annotations)} images')
#     t2 = time.time()
#     print_flush('Time elapsed: %f s.\n' % (t2 - t1))


