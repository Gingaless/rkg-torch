import sqlite3
import myconfig as conf
import shutil
import os
import sys

def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        return 'Successfully clear the directory, {}.'.format(path)
    else:
        raise FileNotFoundError()

def clear_orig():
    return clear_dir(conf.ORIG_SRC_PATH)

def clear_crop(kill_orig=True):
    crop_path =conf.CROPPED_SRC_PATH
    crop_table = conf.Crop.TABLE_NAME
    drop = conf.Crop.DROP_TABLE
    print(clear_dir(crop_path))
    conn = sqlite3.connect(conf.DB_PATH)
    cur = conn.cursor()
    cur.execute(drop.format(crop_table))
    conn.commit()
    print("Successfully drop the table on cropping process.")
    if kill_orig:
        print(clear_orig())

if __name__=='__main__':
    clear_cropQ = True
    if len(sys.argv) > 1:
        if sys.argv[1] == 'N':
            clear_cropQ = False
    clear_crop(kill_orig=clear_cropQ)