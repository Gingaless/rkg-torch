import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from myconfig import FaceDetectionArgs as fd_config
from utils import list_map

#sys.argv[0] is file name.
'''
class FaceArea:

    def distance(x, y):
        assert type(x)==FaceArea and type(y)==FaceArea
        x_cen = x.get_center()
        y_cen = y.get_center()
        return np.linalg.norm(x_cen - y_cen)

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def get_center(self):
        return (x + w//2, y+h//2)

    def get_area(self):
        return w*h

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

    def __array__(self):
        return np.array(self.get_center())
'''

def bgr2rgb(bgr_img):
    return cv2.cvtColor(bgr_img, 4) # cv2.COLOR_BGR2RGB = 4

def get_face_center(face_data):
    (x,y,w,h) = face_data
    x_cen = x + w//2
    y_cen = y + h//2
    return (x_cen, y_cen)

def get_cost_for_ks(x, k_min, k_max):
    cost_info = []
    for i in range(k_min, k_max+1):
        KM = KMeans(n_clusters = i)
        KM.fit(x)
        cost_info.append([len(KM.cluster_centers_), KM.cluster_centers_, KM.inertia_, KM.labels_, KM])
    return cost_info

def elbow_method(data, k_min, k_max):
    assert len(data) > 2
    KMs = []
    for i in range(k_min, k_max+1):
        KM = KMeans(n_clusters=i)
        KM.fit(data)
        KMs.append(KM)
    costs = list(map(lambda x: x.inertia_, KMs))


filename = sys.argv[1]
cascade_file = fd_config.CASCADE_FILE
cascade = cv2.CascadeClassifier(cascade_file)
clahe = cv2.createCLAHE(**fd_config.CLAHE) # Contrast Limited Adaptive Histogram Equalization
img = cv2.imread(filename, 1) # cv2.IMREAD_COLOR = 1
img_ori = img.copy()
print('image size : ', img.shape)
emp = np.zeros(img.shape, dtype=np.uint8)
gray = cv2.cvtColor(img, 6) # cv2.BGR2GRAY = 6
cgray = clahe.apply(gray)
egray = cv2.equalizeHist(gray)
cegray = clahe.apply(egray)
ecgray = cv2.equalizeHist(cgray)

gc_pairs = [(gray, (0,0,255)), (cgray, (0,255,0)), (egray, (255,0,0)), (cegray, (255,0,255)), (ecgray, (0,255,255))]
centers = []
whs = []
xys = []

for gc in gc_pairs:
    g = gc[0]
    c = gc[1]
    faces = cascade.detectMultiScale(g, **fd_config.dMultiScale)
    print(faces, len(faces))
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), c, 8)
        centers.append(get_face_center((x,y,w,h)))
        whs.append(w)
        whs.append(h)



bufs = []



for g, _ in gc_pairs:
    bufs.append(cv2.cvtColor(g, 8)) # cv2.COLOR_GRAY2BGR = 8

gray, cgray, egray, cegray, ecgray = np.array(bufs, dtype = np.uint8)

xaxis_concat1 = np.concatenate([img_ori, img, gray], axis=1)
xaxis_concat2 = np.concatenate([cgray, egray, cegray], axis=1)
xaxis_concat3 = np.concatenate([ecgray, emp, emp], axis=1)
concat_t = np.concatenate([xaxis_concat1, xaxis_concat2, xaxis_concat3], axis=0)
costs = get_cost_for_ks(centers, 1, len(centers))
print('cost_info: \n\n')
for c in costs:
    n_clsts = c[0]
    clstrs = c[1]
    cost = c[2]
    labels = c[3]
    KM = c[4]
    print('n_clusters : ', n_clsts)
    print('clusters : ', clstrs)
    print('cost : ', cost)
    min_of_w_h = np.min(whs)
    print('min of w,h : ', min_of_w_h)
    print('cost per min of w,h ratio : ', cost/min_of_w_h)
    print('labels : ', labels)
    print('predict : ', KM.predict(centers))
    print()
log_costs = np.log(np.array(list(map(lambda c: c[2], costs)), dtype=np.float))
log_decay = [log_costs[i] - log_costs[i-1] for i in range(1,len(costs))]
print('log_decay : ', log_decay)
print('\n')


plt.imshow(bgr2rgb(concat_t), vmin=0, vmax=255)
plt.show()

'''
[original image, image where face detected, gray image,
gray image clahed, gray image equalized Hist,equalize hist and then clahe,
clahe and equalize hist, empty array, empty array].

face detection color :
gray - red
clahe - green
equalize hist - blue
eqaulize hist and then clahe - purple
clahe and then equalize hist - yellow

'''