
import scrapper_manga as scm
import sys


if __name__=='__main__':
    if len(sys.argv) > 1:
        pic_num = int(sys.argv[1])
        scm.login()
        scm.dl_tag('kiana', pic_num, deep_into_manga=True, add_classname_in_path=False)
        scm.save_garage()
    else:
        raise Exception("Enter pic_num.")

