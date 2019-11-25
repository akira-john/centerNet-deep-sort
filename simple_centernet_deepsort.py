import os
import cv2
import math
import argparse
import numpy as np
import pandas as pd
import pickle
import argparse
import sys
import os.path
import dlib
import time
from collections import OrderedDict
from scipy import ndimage
from imutils.video import FPS
import torch.multiprocessing as mp
import datetime


import sys
CENTERNET_PATH = '/content/centerNet-deep-sort/CenterNet/src/lib'
COUNTER_PATH = '/content/People_Counter'
sys.path.append(CENTERNET_PATH)
sys.path.append(COUNTER_PATH)

# import libraries for tracking
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject

#DLIB_USE_CUDA=1
print (dlib.DLIB_USE_CUDA)

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
skip_frames = 15     #No. of frames skipped for next detection

parser = argparse.ArgumentParser(description='Object Detection and Tracking using YOLO in OPENCV')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--outpath')
parser.add_argument('--rotate', type=int, default=0)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--write', type=int, default=0)
parser.add_argument('--night', type=int, default=0)
parser.add_argument('--screenshot', type=int, default=0)
parser.add_argument('--record', type=int, default=1)
parser.add_argument('--outimage', default=None)
parser.add_argument('--finish', type=int, default=-1)
parser.add_argument('--log', default='/content/centerNet-deep-sort/log.txt')
parser.add_argument('--outsize', type = int, default = 600)
parser.add_argument("-s", "--skip_frames", type=int, default=4)
args = parser.parse_args()

#CenterNet
from detectors.detector_factory import detector_factory
from opts import opts


MODEL_PATH = '/content/centerNet-deep-sort/CenterNet/models/ctdet_coco_dla_2x.pth'
ARCH = 'dla_34'

#MODEL_PATH = './CenterNet/models/ctdet_coco_resdcn18.pth'
#ARCH = 'resdcn_18'



TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, ARCH).split(' '))

#vis_thresh
opt.vis_thresh = args.threshold


#input_type
opt.input_type = 'vid'   # for video, 'vid',  for webcam, 'webcam', for ip camera, 'ipcam'

#------------------------------
# for video
opt.vid_path = args.video  #
#------------------------------
# for webcam  (webcam device index is required)
opt.webcam_ind = 0
#------------------------------
# for ipcamera (camera url is required.this is dahua url format)
opt.ipcam_url = 'rtsp://{0}:{1}@IPAddress:554/cam/realmonitor?channel={2}&subtype=1'
# ipcamera camera number
opt.ipcam_no = 8
#------------------------------


from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time


def bbox_to_xywh_cls_conf(bbox_0):
    person = [[1], opt.vis_thresh]
    bicycle = [[2], 0.35]
    car = [[3, 6, 8], 0.5]
    motorcycle = [[4], 0.5]
    # babycar = [[5], 0.5]
    bbox_lis = []

    for ide, thresh in [person, bicycle, car, motorcycle]:
        bbox = []
        for i in ide:
            bbox_1 = bbox_0[i]
            bbox.append(bbox_1)
        bbox = np.concatenate(bbox, axis=0)
        if any(bbox[:, 4] > thresh):

            bbox = bbox[bbox[:, 4] > thresh, :]
            bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
            bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #

            bbox_lis.append((bbox[:, :4], bbox[:, 4]))

        else:
            bbox_lis.append((None, None))
    return bbox_lis


def CenterNet(inQ, outQ):
    det = Detector(opt)
    det.open(opt.vid_path)
    num = 0
    while True:
        num += 1
        ori_im = inQ.get()
        if ori_im == 'fin':
            break
        if args.rotate:
            ori_im = ndimage.rotate(ori_im, 270, reshape=True)
        results = det.detector.run(ori_im)['results']
        bbox_lis = bbox_to_xywh_cls_conf(results)
        outQ.put((ori_im, bbox_lis))
        if num % 100 == 0:
            print('CenterNet : {} / {}'.format(num, det.frames//args.skip_frames))
    outQ.put(['fin', 'fin'])


def adjust(img, alpha=1.0, beta=0.0):
    # 積和演算を行う。
    dst = alpha * img + beta
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(dst, 0, 255).astype(np.uint8)


def get_video(inq2, inq4):
    det = Detector(opt)
    det.open(opt.vid_path)
    frame_no = -1
    num = args.skip_frames
    while det.vdo.grab():
        start_0 = time.time()
        frame_no +=1
        if frame_no == args.finish:
            break
        if frame_no % 100 == 0:
            print('get_video : {} / {}'.format(frame_no, det.frames))
        if frame_no % num == 0:
            inQ = inq4
        elif (frame_no+(num//2)) % num == 0:
            inQ = inq2
        else:
            continue
        _, ori_im = det.vdo.retrieve()
        if not _:
            break
        ori_im = scale_to_width(ori_im, det.im_width)
        if args.night:
            ori_im = adjust(ori_im, alpha=1.2, beta=20)
        if args.screenshot:
            if frame_no % args.screenshot == 0:
                path = os.path.join('./screenshot', str(frame_no) + '_shot.png')
                cv2.imwrite(path, ori_im)
                continue
            else:
                continue
        inQ.put(ori_im)
    PILL = 'fin'
    print(PILL)
    inq2.put(PILL)
    inq4.put(PILL)


class Detector(object):
    def __init__(self, opt):
        self.vdo = cv2.VideoCapture()

        #centerNet detector
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")

        self.write_video = args.write

    def open(self, video_path):

        if opt.input_type == 'webcam':
            self.vdo.open(opt.webcam_ind)

        elif opt.input_type == 'ipcam':
            # load cam key, secret
            with open("cam_secret.txt") as f:
                lines = f.readlines()
                key = lines[0].strip()
                secret = lines[1].strip()

            self.vdo.open(opt.ipcam_url.format(key, secret, opt.ipcam_no))

        # video
        else :
            Width, Height = 1, 1
            assert os.path.isfile(opt.vid_path), "Error: path error"
            self.vdo.open(opt.vid_path)
            Width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            Height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #self.vdo.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
            #self.vdo.set(cv2.CAP_PROP_FRAME_HEIGHT,int(600 * (Height / Width)))

        self.im_width = args.outsize
        self.fps = self.vdo.get(cv2.CAP_PROP_FPS)
        self.im_height = int(self.im_width * (Height / Width))
        self.frames = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT));

        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(args.outpath + '.avi', fourcc, 15, (self.im_width, int(self.im_width * (Height / Width))))
        #return self.vdo.isOpened()


    def detect(self):
        start_00 = time.time()
        DataList = OrderedDict()
        bbox = []
        R_or_L = {'per':[0, 0], 'bic':[0, 0], 'car':[0, 0], 'mot':[0, 0], 'baby': [0, 0]}
        W = self.im_width
        H = self.im_height
        if args.rotate:
            W, H = H, W
        valid_rec = [(W//4, 0), (3*W//4, 0), (W//4, H), (3*W//4, H)]
        if args.screenshot:
            if not os.path.exists('./screenshot'):
                os.mkdir('./screenshot')
        bt = CentroidTracker(maxDisappeared=20, maxDistance=250)
        ct = CentroidTracker(maxDisappeared=3, maxDistance=250)
        mt = CentroidTracker(maxDisappeared=3, maxDistance=250)
        babyt = CentroidTracker(maxDisappeared=30, maxDistance=50)
        [p2, inq2, outq2] = [p4, inq4, outq4] = [0, mp.SimpleQueue(), mp.SimpleQueue()]
        video = mp.Process(
            target=get_video,
            args=(inq2, inq4))
        processes = [video]
        for p, inq, outq in [[p2, inq2, outq2], [p4, inq4, outq4]]:
            p = mp.Process(
                target=CenterNet,
                args=(inq, outq))
            processes.append(p)
        for process in processes:
            process.start()
        ob_dict = {}
        ob_num = 0
        frame_2 = -1
        fin_count = 0
        num = args.skip_frames

        while True:
            start_0 = time.time()
            frame_2 += 1
            if frame_2 % num == 0:
                outQ = outq4
            elif (frame_2+(num//2)) % num == 0:
                outQ = outq2
            else:
                continue
            rects = []
            dlibs = []
            objects = OrderedDict()
            ori_im, bbox_lis = outQ.get()
            if ori_im == 'fin':
                fin_count += 1
                if fin_count == 2:
                    break
                continue

            for c, (bbox_xywh, cls_conf) in enumerate(bbox_lis):
                if c==0 and bbox_xywh is not None:
                    outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_im)
                    if len(outputs) > 0:
                        bbox_xywh = outputs[:, :4]
                        identities = outputs[:, 4]
                        for i, identity in enumerate(identities):
                            if identity in ob_dict.keys():
                                identity = ob_dict[identity]
                            else:
                                ob_dict[identity] = ob_num
                                identity = ob_num
                                ob_num += 1
                            x, y, w, h = bbox_xywh[i]
                            box = (x, y, x+w, y+h)
                            cent = (int(x + (w/2)), int(y + (h/2)))
                            objects[identity] = [cent, box, c]
                elif c != 0:
                        rects = []
                        cl, t = [['bic', bt], ['car', ct], ['mot', mt], ['baby', babyt]][c-1]
                        if bbox_xywh is not None:
                            for i in range(bbox_xywh.shape[0]):
                                x, y, w, h = bbox_xywh[i]
                                rects.append((x, y, x+w, y+h))
                        t_objects = t.update(rects)
                        for ob_id, cent_lis in t_objects.items():
                            if not cent_lis[1] in rects:
                                continue
                            cl_id = cl + str(ob_id)
                            cent, box = cent_lis
                            objects[cl_id] = [cent, box, c]

            frame_info = [frame_2, self.frames, self.fps, W, H]

            # Mark  all the persons in the frame
            count, R_or_L = MarkPeople(ori_im, objects, R_or_L, valid_rec, DataList, frame_info, bbox)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("PersonRight ", R_or_L['per'][0]-R_or_L['bic'][0]),
                ("PersonLeft ", R_or_L['per'][1]-R_or_L['bic'][1]),
                ('BicycleRight', R_or_L['bic'][0]),
                ('BicycleLeft', R_or_L['bic'][1]),
                ('MotRight', R_or_L['mot'][0]),
                ('MotLeft', R_or_L['mot'][1]),
                ('CarRight', R_or_L['car'][0]),
                ('CarLeft', R_or_L['car'][1])
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(ori_im, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            #print(ori_im.shape)

            if args.write:
                cv2.rectangle(ori_im, valid_rec[0], valid_rec[3], (100, 100, 100), 1)
                self.output.write(ori_im)

            end = time.time()

            fps =  1 / (end - start_0)
            avg_fps = frame_2 / (end - start_00)

            if frame_2 % 100 == 0:
                print("{}/{}  centernet time: {:.3f}s, fps: {:.3f}, avg fps : {:.3f}".format(frame_2, self.frames, end - start_00, fps,  avg_fps))

        for x, y in info:
            print(x, y)
        for p in processes:
            print(p, ' - finish')
            p.join()

        return DataList, bbox


# Draw the predicted bounding box
def MarkPeople(frame, objects, R_or_L, valid_rec, DataList, frame_info, bbox):
    count = 0
    de = []
    frames, sum_frames, fps, W, H = frame_info
    now = round(frames/fps, 2)
    per_lis = [(objectID, cent_lis[0]) for objectID, cent_lis in objects.items() if cent_lis[2]==0]
    class_lis = ['per', 'bic', 'car', 'mot', 'baby']
    # loop over the tracked objects
    for (objectID, cent_lis) in objects.items():
        c = cent_lis[2]
        key = class_lis[c]
        centroid = cent_lis[0]
        on_bicycle = False
        bounding = None
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        else:
            to.centroids.append(centroid)
            x_centroids = [x[0] for x in to.centroids]
            x_dist = x_centroids[-1] - x_centroids[0]
            mean = sum(x_centroids) / len(x_centroids)
            to.dirc = 0 if centroid[0]-mean > 0 else 1
            # check to see if the object has been counted or not
            if not to.counted:
                if c==0 and valid_rec[0][0] <= centroid[0] <= valid_rec[3][0] \
                        and valid_rec[0][1] <= centroid[1] <= valid_rec[3][1] and (cent_lis[1][2]-cent_lis[1][0])/2 <= abs(x_dist):           
                    if to.dirc != -1:
                        R_or_L[key][to.dirc] += 1
                        to.counted = True
                    bounding = cent_lis[1]
                    if args.write:
                        cv2.rectangle(frame, (bounding[0], bounding[1]), (bounding[2], bounding[3]), (0, 0, 255), 1)
                elif c==1 and valid_rec[0][0] <= centroid[0] <= valid_rec[3][0]\
                        and valid_rec[0][1] <= centroid[1] <= valid_rec[3][1] and (cent_lis[1][2]-cent_lis[1][0])/2 <= abs(x_dist):
                    cents = []
                    IDs = []
                    x, y = cent_lis[0]
                    h = cent_lis[1][3] - cent_lis[1][1]
                    for o_id, cent in per_lis:
                        rad = math.atan2(y-cent[1], x-cent[0])
                        dist = np.linalg.norm(cent-cent_lis[0])
                        if 15 <= math.degrees(rad) <= 165 and dist <= h:
                            cents.append(cent) 
                            IDs.append(o_id)
                    if len(cents) > 0:
                        m_idx = serch_neighbourhood(cent_lis[0], cents)
                        if args.write:
                            cv2.line(frame,(x, y), (cents[m_idx][0], cents[m_idx][1]), (255,0,0), 2)
                        to_per = trackableObjects.get(IDs[m_idx], None)
                        to_per.cycle += 5
                        to.cycle += 5
                        if to.dirc==to_per.dirc:
                            R_or_L['bic'][to.dirc] += 1
                            to.counted = True
                elif c!=0 and c!=1 and W/5 <= abs(x_dist):
                    if valid_rec[0][0] <= centroid[0] and to.dirc == 0:
                        R_or_L[key][0] += 1
                        to.counted = True
                        bounding = [0, 0, 0, 0]
                    elif valid_rec[3][0] >= centroid[0] and to.dirc == 1:
                        R_or_L[key][1] += 1
                        to.counted = True
                        bounding = [0, 0, 0, 0]
            if to.counted:
                if (centroid[0] > valid_rec[3][0] and to.dirc == 0) or (centroid[0] < valid_rec[0][0] and to.dirc == 1) :
                    if c==0 or c==1:
                        to.counted = False
                        to.cycle = 0
                        to.centroids = [centroid]


        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        color = (0, 255, 0)
        if to.counted:
            color = (0, 0, 255)
        if to.cycle >=3:
            color = (255, 0, 0)
            if to.unique:
                DataList[to.unique][4] = True

        if c!=1:
            total = R_or_L[key][0] + R_or_L[key][1]
            if bounding:
                miny = int(max(0, bounding[1]-3))
                maxy = int(min(H, bounding[3]+3))
                minx = int(max(0, bounding[0]-4))
                maxx = int(min(W, bounding[2]+4))
                frame_b = frame[miny: maxy, minx: maxx] if c==0 else None
                dirc = '>>' if to.dirc == 0 else '<<'
                to.unique = key+str(total)
                DataList[key+str(total)] = [now, objectID, key, dirc, on_bicycle]
                bbox.append(frame_b)
                if frame_b is not None and args.outimage:
                    path = os.path.join(args.outimage, key + str(total) + "_out.jpg")
                    cv2.imwrite(path, frame_b)


        if args.write:
            x_centroids = [x[0] for x in to.centroids]
            mean = sum(x_centroids) / len(x_centroids)
            dirc = ['>>', '<<', '--'][to.dirc]
            text = "{} {} {}".format(objectID, dirc, to.cycle)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            for i in range(min(len(to.centroids), 20)):
                cent = to.centroids[-i]
                cv2.circle(frame, (cent[0], cent[1]), 1, (0, 250, 250), -1)
            # cv2.circle(frame, (centroid[0], centroid[1]), 2, color, -1)
            bounding = cent_lis[1]
            cv2.rectangle(frame, (bounding[0], bounding[1]), (bounding[2], bounding[3]), color, 1)
        count+=1

    # for i in de:
    #     ct.deregister(i)

    return count, R_or_L


def scale_to_width(img, width):
    height = int(img.shape[0] * (width / img.shape[1]))
    return cv2.resize(img, (width, height))


def serch_neighbourhood(p0, ps):
    L = np.array([])
    for i in range(len(ps)):
        L = np.append(L,np.linalg.norm(ps[i]-p0))
    return np.argmin(L)


def record_to_pd(DataList, bbox):
    df= pd.DataFrame(list(DataList.values()))
    columns = ['time', 'objectID', 'class', 'direction', 'on_bicycle']
    df.columns = columns
    df.to_pickle(os.path.join(args.outpath, args.video[-26:-4], 'df_out.pkl'), protocol=2)
    bbox = np.asarray(bbox)
    np.save(os.path.join(args.outpath, args.video[-26:-4], 'df_out.npy'), bbox)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    startMain = time.time()
    count = 0
    trackableObjects = {}
    det = Detector(opt)
    # det.open("D:\CODE\matlab sample code/season 1 episode 4 part 5-6.mp4")
    det.open(opt.vid_path)
    DataList, bbox = det.detect()
    total_time = time.time() - startMain
    if args.record:
        record_to_pd(DataList, bbox)
    print('-----finish-----', total_time)
