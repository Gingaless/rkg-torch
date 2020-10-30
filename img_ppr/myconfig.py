from utils import to_gray, apply_clahe, eqlHist
import numpy as np

_apply_clahe = lambda gray_img : apply_clahe(gray_img, **FaceDetectionArgs.CLAHE)

class FaceDetectionArgs:

    CASCADE_FILE = "lbpcascade_animeface.xml"
    
    PPR_IMG_FUNC_LIST = [
        to_gray,
        lambda img : _apply_clahe(to_gray(img)),
        lambda img : eqlHist(to_gray(img)),
        lambda img : eqlHist(_apply_clahe(to_gray(img))),
        lambda img : _apply_clahe(eqlHist(to_gray(img)))
    ] # image-preprocessing function list for face detection.

    CLAHE = {
        "clipLimit" : 2.0,
        "tileGridSize" : (8,8)
    }

    MIN_FACE_PER_SL_RATIO = 1.0/16 #minium of face per side length ratio.

    dMultiScale = {
        #"scaleFactor" : ,
        #"minNeighbors" : ,
        #"flags" : ,
        #"minSize" : (8,8),
        #"maxSize" : 
    }

    class Elbow:
        THR_IS_ELBOW = 1 # use initial threshold cost ratio to identify elbow.
        ARGMIN_IS_ELBOW = 2 # use argmin of log decay rate to identify elbow.

        INI_THR_COST_RATIO = 1.0 #initial threshold cost ratio when num of faces = 2
        #THR_COST_V_RATIO = 0.25 #threshold cost decay velocity ratio.
        ELBOW_SEARCH_METHOD = 1 # which is used to find elbow?

        KM_KWARGS = {
            'max_iter' : 500
        }
        FACTOR_FUNC = lambda face_rectangle : np.min(face_rectangle[2:])
        DATA_MAP_FUNC = lambda face_rectangle : np.array(face_rectangle[0:2] + face_rectangle[2:]//2)


IMG_FORMATS = ['jpeg', 'gif', 'png', 'jpg']
DB_PATH = 'ppr_log.db'
ORIG_SRC_PATH = '/home/shy/kiana_orig/1'
CROPPED_SRC_PATH = '/home/shy/kiana_cropped/1'

class Crop:

    TABLE_NAME = 'cropped_and_resized'
    TABLE_SCHEMA = "create table if not exists {} \
        (log_id integer primary key autoincrement, \
            src_path text not null, \
            state text not null, \
            save_path text);" 
            # state is state after processed.
            # Table name is cannot parameterized due to the danger of sql injection attack.
    DROP_TABLE = 'drop table if exists {};'
    INSERT_SQL = "insert into {} (src_path, state, save_path) \
        values (:src_path, :state, :save_path);"
    CHECK_DUPPLICATED = 'select src_path from {} where src_path=?;'
    CROP_SIZE = 512
    IMG_HEIGHT_PER_HEAD_RATIO = (1,6) #Above head, (Head), Below Head.
    #IMG_WIDTH_PER_HEAD_RATIO = (2,2) #Left, (Body Width = 1 Head) ,Right.
    
