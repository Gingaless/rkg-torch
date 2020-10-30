
import cv2
import numpy as np
from sklearn.cluster import KMeans
from myconfig import FaceDetectionArgs as config
from utils import cv2_bgr2rgb, list_map, arr_map
import kmeans_elbow

cascade = cv2.CascadeClassifier(config.CASCADE_FILE)
elbw_find_method = config.Elbow.ELBOW_SEARCH_METHOD

def cvt_elbow_search_method(mtd):
    elbw_find_method = mtd

def detect_faces(cv2_img):
    #cascade = cv2.CascadeClassifier(config.CASCADE_FILE)
    ppr_imgs = list_map(lambda f : f(cv2_img), config.PPR_IMG_FUNC_LIST)
    dms_kwargs = config.dMultiScale
    if not 'minSize' in dms_kwargs.keys() and config.MIN_FACE_PER_SL_RATIO > 0:
        min_sl = np.min(cv2_img.shape[:2])
        min_fl = int(min_sl*config.MIN_FACE_PER_SL_RATIO)
        dms_kwargs['minSize'] = (min_fl, min_fl)
    faces = list_map(lambda x : cascade.detectMultiScale(x, **config.dMultiScale), ppr_imgs)
    if () in faces:
        faces = [f for f in faces if f!=()]
    if faces == []:
        return faces
    else:
        faces = np.concatenate(faces)
        KM = find_KM_elbow(faces)
        r = filter_faces(KM, faces)
        return r


def find_KM_elbow(data, **KM_kwargs):
    m = elbw_find_method
    if m==1:
        return kmeans_elbow.find_KM_elbow_by_thr(data, 
        config.Elbow.INI_THR_COST_RATIO, 
        config.Elbow.FACTOR_FUNC, config.Elbow.DATA_MAP_FUNC, 
        **config.Elbow.KM_KWARGS)
    if m==2:
        return kmeans_elbow.find_KM_elbow_by_log_decay_argmin(data, 
        config.Elbow.INI_THR_COST_RATIO, 
        config.Elbow.FACTOR_FUNC, config.Elbow.DATA_MAP_FUNC, 
        **config.Elbow.KM_KWARGS)
    else:
        raise Exception('Undefined elbow-searching method.')
    

def filter_faces(KM, faces):
    labels = KM.predict(faces[:,0:2] + faces[:,2:]//2)
    label_set = set(labels)
    dict_lbl = dict()
    #buf = []
    r=[]
    for lbl in label_set:
        dict_lbl[lbl] = []
    for i in range(len(faces)):
        dict_lbl[labels[i]].append(faces[i])
    #below : select the biggest faces among faces of same labels.
    for key in dict_lbl.keys():
        areas = list_map(lambda face : face[2]*face[3],dict_lbl[key])
        min_idx = np.argmax(areas)
        r.append(tuple(dict_lbl[key][min_idx]))
    #below : remove nested faces in other faces.
    r.sort(key=lambda x : x[2]*x[3],reverse=True)
    remove_nested_faces(r)
    return r

def remove_nested_faces(face_list):
    find_nested = False
    for x1,y1,w1,h1 in face_list:
        for x2, y2, w2, h2 in face_list:
            if (x1 < x2 and y1 < y2 and x1+w1 > x2+w2 and y1+h1>y2+h2):
                face_list.remove((x2,y2,w2,h2))
                find_nested = True
                break
        if find_nested:
            break
    if find_nested:
        remove_nested_faces(face_list)

class NoFaceDetected(Exception):

    def __init__(self):
        super().__init__('No face detected.')


if __name__=='__main__':
    import sys
    from utils import draw_rects_in_img, plt_show_cv2_clr_img
    img_name = sys.argv[1]
    mtd = 1
    if len(sys.argv)>2:
        mtd = int(sys.argv[2])
    cvt_elbow_search_method(mtd)
    img = cv2.imread(img_name, 1)
    faces = detect_faces(img)
    print(faces)
    draw_rects_in_img(img, faces, (0,0,255),10)
    plt_show_cv2_clr_img(img)