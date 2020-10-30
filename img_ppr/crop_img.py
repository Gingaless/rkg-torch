
import numpy as np
import cv2
import sqlite3
from PIL import Image
from myconfig import Crop, ORIG_SRC_PATH, CROPPED_SRC_PATH, IMG_FORMATS, DB_PATH
from utils import cv2_bgr2rgb
from ani_face_detection import detect_faces, NoFaceDetected
import matplotlib.pyplot as plt
import sys
import os


crop_size = Crop.CROP_SIZE
hph_above, hph_below = Crop.IMG_HEIGHT_PER_HEAD_RATIO
#wph_left, wph_right = Crop.IMG_WIDTH_PER_HEAD_RATIO


def get_crop_area(img_size, face_vec):

    h_full, w_full = img_size # img_size is a tuple such as (h, w).
    if not (h_full >= crop_size and w_full >= crop_size):
        raise CropAreaSmallException((h_full, w_full))
    x_face, y_face, w_face, h_face = face_vec

    # get initial cropping interval of y axis.
    h1_crop = np.max([0, y_face - h_face*hph_above])
    h2_crop = np.min([h_full, y_face + h_face*(hph_below + 1)])
    h_crop = h2_crop - h1_crop
    
    # examine whether the size of interval is big enough to crop.
    if not (h_crop >= crop_size and w_full >= crop_size):
        raise CropAreaSmallException(h_crop)

    # When the size of initial cropping interval is so big that
    # it exceeds the full width of image,
    # the size of interval is adjusted to full width of image.
    if h_crop > w_full:
        h2_crop = h1_crop + w_full
        w1_crop = 0
        w2_crop = w_full
    # else, the interval of x axis is adjusted, which is located in the middle.
    else:
        w_center = x_face + (w_face // 2)
        w1_crop = w_center - (h_crop // 2)
        w2_crop = w_center + (h_crop // 2)
        # for the special case that either left or right of cropping interval 
        # gets out of the image. 
        if w1_crop < 0:
            w2_crop = w2_crop - w1_crop
            w1_crop = 0
        elif w2_crop > w_full:
            w1_crop = w1_crop - (w2_crop - w_full)
            w2_crop = w_full
    
    return h1_crop, h2_crop, w1_crop, w2_crop


def crop_image_yield(cv2_img):
    img_size = cv2_img.shape[0:2]
    faces = detect_faces(cv2_img)
    if len(faces)==0:
        yield NoFaceDetected()
    for face in faces:
        h1, h2 ,w1, w2 = 0, 0, 0, 0
        try:
            h1, h2, w1, w2 = get_crop_area(img_size, face)
        except CropAreaSmallException as e:
            yield e
            continue
        h, w = h2 - h1, w2 - w1
        img = np.zeros((h,w,3),dtype=np.uint8)
        img[0:,0:,0:] = cv2_bgr2rgb(cv2_img)[h1:h2,w1:w2,0:]
        yield img


def resize_img_thumb(img_arr):
    pil_img = Image.fromarray(img_arr)
    pil_img.thumbnail((crop_size, crop_size), Image.ANTIALIAS)
    return pil_img

def crop_img_batch(src_folder_path, save_folder_path):
    if not os.path.exists(src_folder_path):
        raise FileNotFoundError(message = 'Image source folder path is not found.')
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # if database table does not exist, create.
    cur.execute(Crop.TABLE_SCHEMA.format(Crop.TABLE_NAME))
    # if save folder path does not exist, create directory.
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    # call source images file from source directory.
    for f in os.listdir(src_folder_path):
        if f.endswith(tuple(IMG_FORMATS)):
            full_src_path = os.path.join(src_folder_path,f)
            # check whether the image have been already processed or not.
            # if having done, write log and pass it.
            if len(cur.execute(Crop.CHECK_DUPPLICATED.format(
                Crop.TABLE_NAME), (full_src_path,)).fetchall()) > 0:
                print('{} have been already processed.\n'.format(full_src_path))
                continue
            img = cv2.imread(full_src_path,1)
            itr = 0
            for cropped in crop_image_yield(img):
                # if cropping function raise exception, write log and pass it.
                if isinstance(cropped, Exception):
                    print('error : {} while {}, {}th cropped.\n'.format(cropped, full_src_path, itr+1))
                    cur.execute(Crop.INSERT_SQL.format(Crop.TABLE_NAME), 
                    {'src_path' : full_src_path, 
                    'state' : 'Fail : {}'.format(type(cropped)),
                    'save_path' : None})
                    conn.commit()
                    continue
                itr+=1
                print('{}, {}th cropped.'.format(full_src_path, itr))
                # image resizing.
                resized = resize_img_thumb(cropped)
                full_save_path = os.path.join(save_folder_path, os.path.splitext(f)[0]+ '_{}'.format(itr) + '.jpg')
                resized.save(full_save_path)
                print('{}, {}th resized and saved as {}.'.format(full_src_path, itr, full_save_path))
                # write success log.
                cur.execute(Crop.INSERT_SQL.format(Crop.TABLE_NAME), 
                    {'src_path' : full_src_path, 
                    'state' : 'Success',
                    'save_path' : full_save_path})
                conn.commit()
                print('complete.\n')
        else:
            continue
    conn.close()


class CropAreaSmallException(Exception):
    def __init__(self, img_size=None):
        if img_size==None:
            super().__init__('the calculated cropping area is too small.')
        else:
            super().__init__('the calculated cropping aread is too small. It is {}.'.format(img_size))


if __name__=='__main__':
    src_path = ORIG_SRC_PATH
    save_path = CROPPED_SRC_PATH
    if len(sys.argv) > 1 and src_path != '_':
        src_path = sys.argv[2]
    if len(sys.argv) > 2:
        save_path = sys.argv[3]
    crop_img_batch(src_path, save_path)



