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
import time
from collections import OrderedDict
from scipy import ndimage
from imutils.video import FPS
from collections import Counter
import torch
import torch.multiprocessing as mp

import random
import datetime
# from datetime import datetime
from collections import Counter
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
from PIL import Image, ImageFont, ImageDraw

import matplotlib.pyplot as plt

import sys
CENTERNET_PATH = 'CenterNet/src/lib'
COUNTER_PATH = '../People_Counter'
ATTRIBUTE_PATH = '/home/dev/Documents/work/pedestrian-attribute-recognition-pytorch'
sys.path.append(CENTERNET_PATH)
sys.path.append(COUNTER_PATH)
sys.path.append(ATTRIBUTE_PATH)

from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

from baseline.model.DeepMAR import DeepMAR_ResNet50
from baseline.utils.utils import str2bool
from baseline.utils.utils import save_ckpt, load_ckpt
from baseline.utils.utils import load_state_dict 
from baseline.utils.utils import set_devices
from baseline.utils.utils import set_seed

# import libraries for tracking
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject

import line_profiler
# pr = line_profiler.LineProfiler()
# pr.add_function(main)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Object Detection and Tracking')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--video_outpath', help='Path to output video file.')
parser.add_argument('--record_outpath', help='Path to output record file.')
parser.add_argument('--rotate', type=int, default=0, help='Use if a video file is rotated')
parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold of person')
parser.add_argument('--write', type=int, default=0, help='Output a video file or not')
parser.add_argument('--image_write', type=int, default=0, help='Output image files or not')
parser.add_argument('--night', type=int, default=0, help='Use if a video file is darker than expected')
parser.add_argument('--screenshot', type=int, default=0, help='Take a screenshot per designated frames')
parser.add_argument('--record', type=int, default=1, help='Output a npy and pkl files ot not')
parser.add_argument('--image_outpath', default=None, help='Path to output pedestrian images')
parser.add_argument('--finish', type=int, default=-1, help='You can finish this process at this frame_num')
parser.add_argument('--attribute_weight_file', type=str)
# parser.add_argument('--log', default='/content/centerNet-deep-sort/log.txt')
parser.add_argument('--outsize', type = int, default = 600, help="Output video's size")
parser.add_argument("-s", "--skip_frames", type=int, default=4, help="Detect per this figure")
args = parser.parse_args()

#CenterNet
from detectors.detector_factory import detector_factory
from opts import opts

# ローカル環境で動かそうとした場合、この設定がないとGPUメモリが足りなくなる
torch.backends.cudnn.enabled = True

MODEL_PATH = 'CenterNet/models/ctdet_coco_dla_2x.pth'
ARCH = 'dla_34'

# MODEL_PATH = './CenterNet/models/ctdet_coco_resdcn18.pth'
# ARCH = 'resdcn_18'

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
# opt.ipcam_url = 'rtsp://{0}:{1}@IPAddress:554/cam/realmonitor?channel={2}&subtype=1'
# ipcamera camera number
opt.ipcam_no = 8
#------------------------------

age_box = ('05', '15', '25', '35', '45', '55', '65', '75')
age_box_2 = ('05', '13', '22', '30', '50', '60', '68', '77')
age_counter = {'05': 0, '15': 0, '25': 0, '35': 0, '45': 0, '55': 0, '65': 0, '75': 0}
gender_counter = {'Male': 0, 'Female': 0}
b_age_counter = {'05': 0, '15': 0, '25': 0, '35': 0, '45': 0, '55': 0, '65': 0, '75': 0}
b_gender_counter = {'Male': 0, 'Female': 0}
# date = datetime.datetime.strptime(args.video[-30:-13], '%Y%m%d_%H-%M-%S')
date = datetime.datetime.now()
place_id = '0001'
place_id = '0001'
device_id = '0001'

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=0)

car_pos = []

# mot_bic_model
mot_model = models.resnet34(pretrained=True)
mot_model.fc =  nn.Linear(in_features=512, out_features=2)
mot_model.load_state_dict(torch.load('/home/dev/Documents/work/mot_bic_11th_100.pt'))
mot_model.to('cuda')
mot_model.eval()
mot_transforms = transforms.Compose([transforms.Resize((224, 224)),
                      transforms.CenterCrop((224, 224)),
                      transforms.ToTensor(),
                      transforms.Normalize((0.444, 0.439, 0.434), (0.111, 0.099, 0.09))])

# attribute_recognition
model_kwargs = dict()
model_kwargs['num_att'] = 9
model_kwargs['last_conv_stride'] = 2

model = DeepMAR_ResNet50(**model_kwargs)
model.classifier.out_features=9

map_location = (lambda storage, loc:storage)
ckpt = torch.load(args.attribute_weight_file, map_location=map_location)
new_state_dict = OrderedDict()
# for k, v in ckpt['state_dicts'][0].items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
model.load_state_dict(ckpt['state_dicts'][0])
model.cuda()
model.eval()

# CenterNetから出力されたBBデータを各classのThresholdに基づいて選別し、xyxy形式からwywh形式にした上で返す
def bbox_to_xywh_cls_conf(bbox_0):
    # 各クラスのThreshold
    person = [[1], opt.vis_thresh]
    bicycle = [[2, 4], 0.30]
    car = [[3, 6, 8], 0.5]
    # motorcycle = [[4], 0.5]
    # babycar = [[5], 0.5]
    bbox_lis = []

    for ide, thresh in [person, bicycle, car]:
        bbox = []
        for i in ide:
            bbox_1 = bbox_0[i]
            bbox.append(bbox_1)
        bbox = np.concatenate(bbox, axis=0)
        if any(bbox[:, 4] > thresh):
            bbox = bbox[bbox[:, 4] > thresh, :]
            try:
                cv2.cvtColor(bbox, cv2.COLOR_BGR2RGB)
            except:
                print(bbox.shape, bbox)
                bbox_lis.append((None, None))
                continue
            bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
            bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #

            bbox_lis.append((bbox[:, :4], bbox[:, 4]))
        else:
            bbox_lis.append((None, None))
    return bbox_lis

# 画像を明るく、コントラストを強くする
def adjust(img, alpha=1.0, beta=0.0):
    # 積和演算を行う。
    dst = alpha * img + beta
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(dst, 0, 255).astype(np.uint8)


class Detector(object):

    def __init__(self, opt):
        self.vdo = cv2.VideoCapture()
        #centerNet detector
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        self.write_video = args.write

    def open(self, video_path):
        Width, Height = 1, 1
        print(opt.vid_path)
        assert os.path.isfile(opt.vid_path), "Error: path error"
        self.vdo.open(opt.vid_path)
        Width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        Height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #self.vdo.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        #self.vdo.set(cv2.CAP_PROP_FRAME_HEIGHT,int(600 * (Height / Width)))
        # 指定された出力時の映像サイズを横幅に適用し、縦幅はアスペクト比を維持できるように設定する
        self.im_width = args.outsize
        self.im_height = int(self.im_width * (Height / Width))
        self.fps = self.vdo.get(cv2.CAP_PROP_FPS)
        self.frames = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT));
        # ビデオを出力する場合のPathや方式の指定
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(args.video_outpath + '.avi', fourcc, 15, (self.im_width, int(self.im_width * (Height / Width))))
        #return self.vdo.isOpened()

    @profile
    def detect(self):
        start_00 = time.time()
        DataList = OrderedDict()
        bbox = []
        R_or_L = {'per':[0, 0], 'bic':[0, 0], 'car':[0, 0], 'mot':[0, 0], 'baby': [0, 0]}
        W = self.im_width
        H = self.im_height
        if args.rotate:
            W, H = H, W
        if args.screenshot:
            if not os.path.exists('./screenshot'):
                os.mkdir('./screenshot')
        # カウントを実際に行う領域
        valid_rec = [(W//8, 0), (7*W//8, 0), (W//8, H), (7*W//8, H)]
        count_rec = [(4*W//9, 0), (5*W//9, 0), (4*W//9, H), (5*W//9, H)]
        # 人以外のclassはDeepSortを使えないため、簡易版のTrackerを用意する
        bt = CentroidTracker(maxDisappeared=20, maxDistance=250)
        ct = CentroidTracker(maxDisappeared=20, maxDistance=250)
        mt = CentroidTracker(maxDisappeared=20, maxDistance=250)
        babyt = CentroidTracker(maxDisappeared=30, maxDistance=50)

        ob_dict = {}
        ob_num = 0
        frame_2 = -2
        Break_flg = False

        while True:
            frame_lis = []
            ori_im_lis = []
            for i in range(8):
                if frame_2==args.finish:
                    Break_flg = True
                    break
                _, ori_im = det.vdo.read()
                if not _:
                    Break_flg = True
                    break
                if i%2==0:
                    ori_im = scale_to_width(ori_im, W)
                    ori_im_lis.append(ori_im)
                    frame_lis.append(np.array(ori_im))
            # print(len(ori_im_lis), frame_2)
            # start_0 = time.time()
            objects = OrderedDict()
            results = det.detector.run(np.array(frame_lis))
            # torch.cuda.empty_cache()
            # {c=0: person, c=1: bicycle, c=2: car, c=3: motorbike}
            for ori_im, result in zip(ori_im_lis, results):
                frame_2 += 2
                bbox_lis = bbox_to_xywh_cls_conf(result['results'])
                for c, (bbox_xywh, cls_conf) in enumerate(bbox_lis):
                    if c==0 and bbox_xywh is not None:
                        # personのみDeepsortにより精度の高いTrackingを行う
                        outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_im)
                        if len(outputs) > 0:
                            bbox_xywh = outputs[:, :4]
                            identities = outputs[:, 4]
                            for i, identity in enumerate(identities):
                                # ObjectIDを連番にするためにDictを用いて処理を行っている
                                if identity in ob_dict.keys():
                                    identity = ob_dict[identity]
                                else:
                                    ob_dict[identity] = ob_num
                                    identity = ob_num
                                    ob_num += 1
                                x, y, w, h = bbox_xywh[i]
                                box = (x, y, x+w, y+h)
                                cent = (int(x + (w/2)), int(y + (h/2)))
                                # Centroidの座標、BoundingBoxの座標、classを辞書型のobjectsに格納
                                objects[identity] = [cent, box, c]
                    # person以外のclassは簡易型のCentroidTrackerに登録
                    elif c != 0:
                        rects = []
                        class_name, t = [['bic', bt], ['car', ct], ['mot', mt], ['baby', babyt]][c-1]
                        if bbox_xywh is not None:
                            if c==2:
                                car_area = 0
                                for bbox in bbox_xywh:
                                    if bbox[2]*bbox[3] >= car_area:
                                        largest_car = bbox
                                        car_area = bbox[2]*bbox[3]
                                bbox_xywh = largest_car.reshape(1, 4)
                            for i in range(bbox_xywh.shape[0]):
                                x, y, w, h = bbox_xywh[i]
                                rects.append((x, y, x+w, y+h))
                        t_objects = t.update(rects)
                        for ob_id, cent_lis in t_objects.items():
                            if not cent_lis[1] in rects:
                                continue
                            cl_id = class_name + str(ob_id)
                            cent, box = cent_lis
                            objects[cl_id] = [cent, box, c]

                frame_info = [frame_2, self.frames, self.fps, W, H]

                # count、marking、その他処理を行う
                R_or_L = MarkPeople(ori_im, objects, R_or_L, valid_rec, count_rec, DataList, frame_info, bbox)

                info = [
                    ("PersonRight ", R_or_L['per'][0]-R_or_L['bic'][0]),
                    ("PersonLeft ", R_or_L['per'][1]-R_or_L['bic'][1]),
                    ('BicycleRight', R_or_L['bic'][0]-R_or_L['mot'][0]),
                    ('BicycleLeft', R_or_L['bic'][1]-R_or_L['mot'][1]),
                    ('MotRight', R_or_L['mot'][0]),
                    ('MotLeft', R_or_L['mot'][1]),
                    ('CarRight', R_or_L['car'][0]),
                    ('CarLeft', R_or_L['car'][1])
                ]

                if args.write:
                    # loop over the info tuples and draw them on our frame
                    for (i, (k, v)) in enumerate(info):
                        text = "{}: {}".format(k, v)
                        cv2.putText(ori_im, text, (10, H - ((i * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.rectangle(ori_im, valid_rec[0], valid_rec[3], (100, 100, 100), 1)
                    cv2.rectangle(ori_im, count_rec[0], count_rec[3], (100, 100, 100), 1)
                    self.output.write(ori_im)

                end = time.time()

                # fps =  1 / (end - start_0)
                avg_fps = frame_2 / (end - start_00)

                if frame_2 % 1000 == 0:
                    print("{}/{} avg fps : {:.3f}".format(frame_2, self.frames, avg_fps))
                if frame_2 % 5000 == 0:
                    print(age_counter)
                    print(gender_counter)

            if Break_flg:
                break

        for x, y in info:
            print(x, y)
        return DataList, bbox


# Draw the predicted bounding box
def MarkPeople(frame, objects, R_or_L, valid_rec, count_rec, DataList, frame_info, bbox):
    frames, sum_frames, fps, W, H = frame_info
    now = round(frames/fps, 1)
    per_lis = [(objectID, cent_lis[0]) for objectID, cent_lis in objects.items() if cent_lis[2]==0]
    class_lis = ['per', 'bic', 'car', 'mot', 'baby']
    draw_list = []
    # loop over the tracked objects
    for (objectID, cent_lis) in objects.items():
        age = gender = None
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
            x_half = np.array(x_centroids) - W//2
            x_dist = x_centroids[-1] - x_centroids[0]
            mean = sum(x_centroids) / len(x_centroids)
            to.dirc = 0 if centroid[0]-mean > 0 else 1
            # check to see if the object has been counted or not
            if not to.counted:
                if c==0 and count_or_not(centroid, x_centroids, valid_rec, count_rec, W):           
                    if to.dirc != -1 and len(to.info)>0:
                        R_or_L[key][to.dirc] += 1
                        to.counted = True
                        bounding = cent_lis[1]
                        to.info.append(conclude_age_gender(to.info))

                elif c==1 and valid_rec[0][0] <= centroid[0] <= valid_rec[3][0]\
                        and valid_rec[0][1] <= centroid[1] <= valid_rec[3][1] and W/8 <= abs(x_dist):
                    cents = []
                    IDs = []
                    x, y = cent_lis[0]
                    h = cent_lis[1][3] - cent_lis[1][1]
                    # バイクかどうかを判定
                    if to.is_mot is None:
                        miny, maxy, minx, maxx = bbox_to_image(cent_lis[1], H, W)
                        frame_b = frame[miny: maxy, minx: maxx]
                        # まれに（何故か）色を変換できない場合があるが、プロセスが止まらないように例外処理
                        try:
                            file = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
                        except:
                            print('bicycle :', frame_b.shape)
                            continue
                        img = Image.fromarray(np.uint8(file))
                        img_trans = mot_transforms(img) 
                        img_trans = torch.unsqueeze(img_trans, dim=0)
                        img_var = Variable(img_trans).cuda()
                        score = mot_model(img_var).data.cpu().numpy()
                        if np.argmax(score)==1:
                            key = 'mot'
                            to.is_mot = True
                            to.counted = True
                            bounding = [0, 0, 0, 0]
                            R_or_L[key][to.dirc] += 1
                            ########
                            # cv2.imwrite('./mot_{}.png'.format(frames), frame_b)
                        else:
                            to.is_mot = False
                    # このフレーム中にいるperのリストを呼び出し、それぞれとの距離および角度を比較
                    for o_id, cent in per_lis:
                        rad = math.atan2(y-cent[1], x-cent[0])
                        dist = np.linalg.norm(cent-cent_lis[0])
                        if 15 <= math.degrees(rad) <= 165 and dist <= h:
                            cents.append(cent) 
                            IDs.append(o_id)
                    # 条件を満たす候補のうち、もっとも距離が近いperを採用
                    if len(cents) > 0:
                        m_idx = serch_neighbourhood(cent_lis[0], cents)
                        if args.write:
                            cv2.line(frame,(x, y), (cents[m_idx][0], cents[m_idx][1]), (255,0,0), 2)
                        to_per = trackableObjects.get(IDs[m_idx], None)
                        if to_per is not None:
                            if to.dirc==to_per.dirc:
                                to_per.cycle += 5
                                to.cycle += 5
                                R_or_L['bic'][to.dirc] += 1
                                to.counted = True

                elif c==2 and W/8 <= abs(x_dist):
                    if valid_rec[0][0] <= centroid[0] and to.dirc == 0:
                        R_or_L[key][0] += 1
                        to.counted = True
                        bounding = [0, 0, 0, 0]
                    elif valid_rec[3][0] >= centroid[0] and to.dirc == 1:
                        R_or_L[key][1] += 1
                        to.counted = True
                        bounding = [0, 0, 0, 0]

            else:
                if (centroid[0] > valid_rec[3][0] and to.dirc == 0) or (centroid[0] < valid_rec[0][0] and to.dirc == 1) :
                    if c==0 or c==1:
                        to.counted = False
                        to.cycle = 0
                        to.is_mot = None
                        to.info=[]
                        to.centroids = [centroid]


        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        color = [0, 255, 0]
        color_2 = [0, 255, 0]
        if to.counted:
            color = [0, 0, 255]
        if to.cycle >=3:
            color = [255, 0, 0]
            if to.unique and DataList[to.unique][6]==False:
                DataList[to.unique][6] = True
                b_gender_counter[DataList[to.unique][7]] += 1
                gender_counter[DataList[to.unique][7]] -= 1
                b_age_counter[DataList[to.unique][8]] += 1
                age_counter[DataList[to.unique][8]] -= 1
        if to.is_mot:
            color = [255, 0, 255]

        if len(to.info)>0:
            age, gender, score = to.info[-1]
            color_2 = [120, 120, 120]
            i = 2 if gender == 'Female' else 0
            color_2[i] = 255-int(age)

        total = R_or_L[key][0] + R_or_L[key][1]

        if bounding:
            miny, maxy, minx, maxx = bbox_to_image(bounding, H, W)
            frame_b = frame[miny: maxy, minx: maxx] if c==0 else None
            dirc = '>>' if to.dirc == 0 else '<<'
            to.unique = key+str(total) if c==0 else None
            now_datetime = date + datetime.timedelta(seconds=now)
            now_date = now_datetime.strftime('%Y%m%d')
            now_time = now_datetime.strftime('%H%M%S.%f')
            if age is not None:
                age_counter[age] += 1
                gender_counter[gender] += 1
            DataList[key+str(total)] = [place_id, device_id, str(now_date), str(now_time[:-5]), key, dirc, on_bicycle, gender, age]
            if args.image_write:
                if c==0:
                    folder_path = os.path.join(args.image_outpath, gender, age)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    path = os.path.join(folder_path, args.video[-26:-4] + '_' + str(total) + ".png")
                    cv2.imwrite(path, frame_b)

        if frames%12==0 and c==0 and not to.counted:
            miny, maxy, minx, maxx = bbox_to_image(cent_lis[1], H, W)
            frame_c = frame[miny: maxy, minx: maxx]
            try:
                file = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
            except:
                # import traceback
                # traceback.print_exc()
                print('person', frame_c.shape, [miny, maxy, minx, maxx])
                continue
            img = Image.fromarray(np.uint8(file))
            img_trans = test_transform(img) 
            img_trans = torch.unsqueeze(img_trans, dim=0)
            img_var = Variable(img_trans).cuda()
            score = model(img_var).data
            age_score = softmax(score[0, 1:9]).cpu().numpy()*np.array([1.2, 2.0, 1.25, 1.25, 0.99, 0.99, 0.99, 1.0])
            gender_score = score[0, 0].cpu().numpy()
            ages = np.asarray([age_score[x] for x in range(0, 8)])*np.array([1.2, 2.0, 1.25, 1.25, 0.99, 0.99, 0.99, 1.0])
            # ages = np.where(ages>=ages[np.argsort(ages)[-3]], ages, 0)
            pred_age = sum(ages[np.where(ages>0)]*np.array([float(x) for x in age_box_2])[np.where(ages>0)])/sum(ages[ages>0]) if sum(ages[ages>0])!=0 else 0
            age_index = np.abs(np.asarray([float(x) for x in age_box]) - pred_age).argmin()
            age = age_box[age_index]
            gender = 'Male' if gender_score<=0 else 'Female'
            total_score = np.hstack([gender_score, age_score])
            to.info.append([age, gender, total_score])

        if args.write:
            x_centroids = [x[0] for x in to.centroids]
            mean = sum(x_centroids) / len(x_centroids)
            dirc = ['>>', '<<', '--'][to.dirc]
            if len(to.info) > 0:
                age, gender, score = to.info[-1]
            else:
                age, gender = '_', '_'
            text = "{} {} {}".format(dirc, age, gender)
            color_2 = color if c!=0 else color_2
            bounding = cent_lis[1]

            # cv2.rectangle(frame, (bounding[0], bounding[1]), (bounding[2], max(0, int(bounding[1] - 11))), color_2, -1)
            # cv2.putText(frame, text, (bounding[0], int(bounding[1])),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            draw_list.append([bounding, text, color, color_2, to.counted, to.centroids])

    if args.write:
        for bounding, text, color, color_2, counted, centroids in draw_list:
            draw_bbox(frame, bounding, text, color_2, counted)
            for i in range(min(len(centroids), 20)):
                cent = centroids[-i]
                cv2.circle(frame, (cent[0], cent[1]), 2, color, -1)
            cv2.rectangle(frame, (bounding[0], bounding[1]), (bounding[2], bounding[3]), color_2, 1)

    return R_or_L


def scale_to_width(img, width):
    height = int(img.shape[0] * (width / img.shape[1]))
    return cv2.resize(img, (width, height))


def count_or_not(centroid, x_centroids, valid_rec, count_rec, W):
    if not (valid_rec[0][0] <= centroid[0] <= valid_rec[3][0] and valid_rec[0][1] <= centroid[1] <= valid_rec[3][1]):
        return False
    left_line = np.array(x_centroids)-count_rec[0][0]
    right_line = np.array(x_centroids)-count_rec[3][0]
    for line in [left_line, right_line]:
        if any(line>0) and any(line<0) and W/10 <= abs(x_centroids[-1]-x_centroids[0]):
            return True
    return False


def bbox_to_image(bounding, H, W):
    miny = int(max(0, bounding[1]-3))
    maxy = int(min(H, bounding[3]+3))
    minx = int(max(0, bounding[0]-2))
    maxx = int(min(W, bounding[2]+2))
    return miny, maxy, minx, maxx


def conclude_age_gender(lis):
    ages = np.array([0.0]*8)
    genders = np.array([0.0])
    score = np.array([0]*9)
    i = 0
    age_count = []
    gender_count = []
    for recent_age, gender, score in lis:
        i += 1
        if len(lis)-i>10:
            continue
        ages += score[1:9]
        # ages_array = np.asarray([score[x] for x in range(1, 9)])
        # ages += np.where(ages_array>0.25, ages_array, 0)
        genders += score[0]
        # age_count.append(age)
        # gender_count.append(gender)
    # ages *= np.array([1., 0.50, 0.33, 0.33, 0.33, 0.33, 0.33, 0.49])
    # ages = np.where(np.array(ages)>=ages[np.argsort(ages)[-3]], ages, 0)
    if np.argmax(ages)==0:
        pred_age = 5
    # elif np.argmax(ages)==1:
    #     pred_age = (15*ages[1]+25*ages[2])/(ages[1]+ages[2])
    elif np.argmax(ages)==7:
        pred_age = (65*ages[6]+75*ages[7])/(ages[6]+ages[7])
    else:
        # n_ages = np.zeros(8)
        # for i in [np.argmax(ages)-1, np.argmax(ages), np.argmax(ages)+1]:
        #     n_ages[i] = ages[i]
        n_ages = np.where(np.array(ages)>=ages[np.argsort(ages)[-3]], ages, 0)
        pred_age = sum(np.asarray([float(x) for x in age_box_2])*n_ages)/sum(n_ages)
    # pred_age = sum(ages[np.where(ages>0)]*np.array([float(x) for x in age_box_2])[np.where(ages>0)])/sum(ages[ages>0]) if sum(ages[ages>0])!=0 else 0
    # pred_age = sum(ages*np.array([float(x) for x in age_box]))/sum(ages) if sum(ages)!=0 else 0
    age_index = np.abs(np.asarray([float(x) for x in age_box]) - pred_age).argmin()
    age = age_box[age_index] if pred_age!=0 else recent_age
    # age = Counter(age_count).most_common()[0][0]
    gender = 'Male' if genders[0]<=0 else 'Female'
    # gender = Counter(gender_count).most_common()[0][0]
    return [age, gender, score]
    # age_counter = Counter(ages)
    # gender_counter = Counter(genders)
    # return [age_counter.most_common()[0][0], gender_counter.most_common()[0][0]]


def draw_bbox(img, box, label, color, counted=False, offset=(0,0)):
    '''
        draw box of an id
    '''
    x1,y1,x2,y2 = [int(i+offset[idx%2]) for idx,i in enumerate(box)]
    font_size = 1.2
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size , 1)[0]
    # cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
    cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
    text_color = [255, 255, 255] if counted else [180, 180, 180]
    cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, font_size, text_color, 1)
    return img


def serch_neighbourhood(p0, ps):
    L = np.array([])
    for i in range(len(ps)):
        L = np.append(L,np.linalg.norm(ps[i]-p0))
    return np.argmin(L)


def record_to_pd(DataList, bbox):
    df= pd.DataFrame(list(DataList.values()))
    columns = ['place_id', 'device_id', 'date', 'time', 'class', 'direction', 'on_bicycle', 'gender', 'age']
    df.columns = columns
    for column in ['place_id', 'device_id']:
        df[column] = df[column].str.zfill(4)
    df.to_csv(os.path.join(args.record_outpath, args.video[-30:-13]+'.csv'), index=False, encoding='utf-8')
    # df.to_pickle(os.path.join(args.record_outpath, args.video[-26:-4]+'df_out.pkl'), protocol=2)
    # bbox = np.asarray(bbox)
    # np.save(os.path.join(args.record_outpath, args.video[-26:-4]+'df_out.npy'), bbox)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print('RuntimeError')
    startMain = time.time()
    count = 0
    trackableObjects = {}
    det = Detector(opt)
    # det.open("D:\CODE\matlab sample code/season 1 episode 4 part 5-6.mp4")
    det.open(opt.vid_path)
    DataList, bbox = det.detect()
    total_time = time.time() - startMain
    print(age_counter)
    print(gender_counter)
    print(b_gender_counter)

    # plt.suptitle(os.path.basename(args.attribute_weight_file)[:-4], fontsize=10)
    # plt.subplot(1, 2, 1)
    # sorted_ages = sorted(list(age_counter.keys()))
    # plt.bar(sorted_ages, [age_counter[x] for x in sorted_ages])
    # plt.subplot(1, 2, 2)
    # sorted_gender = sorted(list(gender_counter.keys()))
    # plt.bar(sorted_gender, [gender_counter[x] for x in sorted_gender])
    # plt.savefig('../{}'.format(os.path.basename(args.video)[:-4]))

    if args.record:
        record_to_pd(DataList, bbox)
    print('-----finish-----', total_time)


