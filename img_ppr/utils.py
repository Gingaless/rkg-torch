
import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv2_bgr2rgb(bgr_img):
    return cv2.cvtColor(bgr_img, 4) # cv2.COLOR_BGR2RGB = 4

def apply_clahe(gray_img, **clahe_kwargs):
    clahe = cv2.createCLAHE(**clahe_kwargs)
    return clahe.apply(gray_img)

def to_gray(cv2_img):
    return cv2.cvtColor(cv2_img, 6) # cv2.BGR2GRAY = 6

def eqlHist(gray_img):
    return cv2.equalizeHist(gray_img)

def list_map(*args, **kwargs):
    return list(map(*args, **kwargs))

def arr_map(*args, dtype=np.float, **kwargs):
    return np.array(list_map(*args, **kwargs), dtype=dtype)

#draw rectangles in image according to retangle-presenting vectors.
def draw_rects_in_img(img, rect_presenting_vec, *rect_args, **rect_kwargs):
    assert len(np.shape(rect_presenting_vec))==2 and np.shape(rect_presenting_vec)[-1]==4
    #rectangle_presenting_vectors are like [(x,y,w,h)].
    for (x,y,w,h) in rect_presenting_vec:
        cv2.rectangle(img, (x,y), (x+w,y+h), *rect_args, **rect_kwargs)

def plt_show_cv2_clr_img(cv2_color_img):
    img = cv2_bgr2rgb(cv2_color_img)
    plt.imshow(img, vmin=0, vmax=255)
    plt.show()