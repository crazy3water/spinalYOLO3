# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
import re
import cv2
from timeit import default_timer as timer
import readyData

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw,  ImageFilter

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

class YOLO(object):
    _defaults = {
        "model_path": 'logs/000/trained_weights1.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "score" : 0.3,#0.3
        "iou" : 0.45, #0.45
        "model_image_size" : (256,256),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        mean = np.mean(image)
        std = np.mean(np.square(image - mean))
        image_data = (image_data - mean) / np.sqrt(std)

        # image_data = np.where(image_data < 0, 0, image_data)

        # image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        points = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            # print("the number of thickness:",thickness)
            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline="red")#self.colors[c])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     outline = "green")#self.colors[c])
            xCenter,yCenter = int((left+right)/2),int((top+bottom)/2)
            points.append([label.split(' ')[0],xCenter,yCenter])

            draw.point(xy=(xCenter,yCenter), fill="red")
            draw.text(text_origin, label, fill="green", font=font) #[0,0,0]
            del draw

        end = timer()
        # print("time:",end - start)
        return image,points

    def close_session(self):
        # self.sess.close()
        del self.yolo_model

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image,_ = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
            image = image.convert("RGB")
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image,points = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

def changeImage(image):
    m,n = image.size
    box = [int(m*0.1),int(n*0.1),int(m*0.9),int(n*0.9)]
    newImage = image.crop(box)
    newImage = newImage.resize([m,n])
    return newImage

def detect_imgs(yolo,testpath):
    f = open(testpath,'r')
    fstep1 = open("resultStep1.txt",'w')
    testdir = f.readlines()
    for imgdir in testdir:
        img = imgdir.split()[0]
        image = Image.open(img)
        image = image.convert("RGB")

        r_image, points = yolo.detect_image(image)
        fstep1.write(img)
        for point in points:
            fstep1.write(" "+point[0]+","+str(point[1])+","+str(point[2]))
        fstep1.write('\n')

        #将原图片进行裁剪再放缩回来，进行重新检测
        # image = changeImage(image)
        # r_image, points = yolo.detect_image(image)
        # fstep1.write(img)
        # for point in points:
        #     fstep1.write(" "+point[0]+","+str(point[1])+","+str(point[2]))
        # fstep1.write('\n')

    yolo.close_session()



def cv2Testjpg(txtPath,testjpg):
    f = open(txtPath,'r')
    data = f.readlines()
    for jgpPath in os.listdir(testjpg):
        img = cv2.imread(os.path.join(testjpg,jgpPath))
        print(jgpPath)
        for line in data:
            sl = line.split()
            if sl[0].split('/')[-1].split('_')[0] == jgpPath.split('/')[-1].split('_')[0]:
                for l in sl[1:]:
                    kind,x,y = l.split(',')
                    img = cv2.circle(img,(int(x),int(y)),radius=3,color=[0,0,255],thickness=-1)
        cv2.imshow(" ",img)
        cv2.waitKey(0)
    f.close()

from readyData import dicom_metainfo,dicom2array
import json
import matplotlib.pyplot as plt

def step1(dataPath, jsondata, truejsonPath=None):
    '''
    将 图片 与 锚点对应
    :return:
    '''
    image = {}
    count = 0
    study = os.listdir(dataPath)
    for studyI in study:
        Impath = os.listdir(os.path.join(dataPath, studyI))
        # print("study:", studyI)
        for Impathi in Impath:
            try:
                studyUid, seriesUid, instanceUid,desc,idex = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi),
                                                                  ['0020|000d', '0020|000e', '0008|0018','0008|103e',"0020|0013"])
                for studyid in jsondata:
                    if studyUid == studyid["studyUid"] and seriesUid == studyid["data"][0][
                        "seriesUid"] and instanceUid == studyid["data"][0]["instanceUid"]:
                        count += 1
                        image[studyI + "_" + Impathi] = dicom2array(
                            os.path.join(os.path.join(dataPath, studyI), Impathi))

                        # 在yolo.py中运行watchTest打开注释
                        img = dicom2array(os.path.join(os.path.join(dataPath, studyI), Impathi))
                        # if studyI == seestudy:
                        plt.figure()
                        plt.title(studyI + "_" + Impathi+"\t"+desc+'\t'+idex)
                        plt.imshow(img)

                        if truejsonPath != None:
                            for studyid1 in truejsonPath:
                                if studyUid == studyid1["studyUid"] and seriesUid == studyid1["data"][0][
                                    "seriesUid"] and instanceUid == studyid1["data"][0]["instanceUid"]:
                                    truePoints = studyid1["data"][0]["annotation"][0]["data"]["point"]
                                    for point in truePoints:
                                        x, y = point['coord']
                                        print("True zIndex:", point["zIndex"])
                                        tag = point['tag']
                                        loc = tag['identification']

                                        if "disc" in tag.keys():
                                            k = tag["disc"]
                                        else:
                                            k = tag["vertebra"]
                                        # if studyI == seestudy:
                                        plt.scatter(x, y, s=12, c='g')
                                        plt.text(x=x + 5, y=y-5, s=loc + '**' + k)

                        for point in studyid["data"][0]["annotation"][0]["data"]["point"]:
                            x, y = point['coord']
                            # print("zIndex:",point["zIndex"] ," red")
                            tag = point['tag']
                            loc = tag['identification']

                            if "disc" in tag.keys():
                                k = tag["disc"]
                            else:
                                k = tag["vertebra"]
                            # if studyI == seestudy:
                            plt.scatter(x, y,s=6,c='r')
                            plt.text(x=x + 5, y=y, s=loc+'--'+k)
                        # if studyI == seestudy:
                        plt.show()
            except:
                print("文件错误：", os.path.join(os.path.join(dataPath, studyI), Impathi))

def watchTest(dataPath,jsonPath):
    with open(jsonPath,"r",encoding="utf-8") as f:
        jsonTarge = json.loads(f.read())
    f.close()
    step1(dataPath,jsondata=jsonTarge)

def watchTest1(dataPath,jsonPath,truejsonPath):
    with open(jsonPath,"r",encoding="utf-8") as f:
        jsonTarge = json.loads(f.read())

    with open(truejsonPath,"r",encoding="utf-8") as f:
        truejsonTarge = json.loads(f.read())

    step1(dataPath,jsondata=jsonTarge,truejsonPath=truejsonTarge)

def metric_json(metricjson,truejson):
    with open(metricjson,"r",encoding="utf-8") as f:
        metricjsonTarge = json.loads(f.read())

    with open(truejson,"r",encoding="utf-8") as f:
        truejsonTarge = json.loads(f.read())

    TP = 0 #在目标内且 如果分类正确
    FP = 0 #在目标内且 分类错误 and 如果预测的点不落在任何标注点
    FN = 0 #标注点没有被任何预测点正确命中
    mm6c = 0
    mm8c = 0
    mmc = 0
    for metrictarget in metricjsonTarge:
        metricstudyUid = metrictarget["studyUid"]
        for truetarget in truejsonTarge:
            if metricstudyUid == truetarget["studyUid"]:#找到
                truepoints = truetarget["data"][0]["annotation"][0]["data"]["point"]
                metricpoints = metrictarget["data"][0]["annotation"][0]["data"]["point"]
                for truepoint in truepoints:
                    mmc += 1
                    xt, yt = list(map(int, truepoint["coord"]))
                    tag = truepoint["tag"]
                    if "disc" in tag.keys():
                        ct = tag["disc"]
                    else:
                        ct = tag["vertebra"]
                    for index,metricpoint in enumerate(metricpoints):
                        x,y = list(map(int,metricpoint["coord"]))
                        tag = metricpoint["tag"]
                        if "disc" in tag.keys():
                            c = tag["disc"]
                        else:
                            c = tag["vertebra"]
                        mm = 6
                        mm8 = 8
                        if x<=xt+mm and x >= xt-mm and y<=yt+mm and y >= yt-mm: #在标注点内
                            mm6c += 1
                            if ct == c:
                                TP += 1
                            else:
                                FP += 1
                            break
                        elif x<=xt+mm8 and x >= xt-mm8 and y<=yt+mm8 and y >= yt-mm8: #在标注点内
                            mm8c += 1
                            print("8mm内，但是没有在6mm内")
                        if index == len(metricpoints)-1:
                            # print("该预测点没有被命中！")
                            FN += 1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print("TP/(TP+FP):",precision,"TP/总定位:",TP/mmc,recall)

def metricDetect(yolo,truejson,testFile):
    with open(truejson,"r",encoding="utf-8") as f:
        truejsonTarge = json.loads(f.read())

    from classifycation import CNN

    sliceResize = [48, 32]
    cnn_disc = CNN(sliceResize)
    model_disc = cnn_disc.loadTestdisc()
    cnn_discv5 = CNN(sliceResize)
    model_discv5 = cnn_discv5.loadTestdiscv5()
    cnn_vertebra = CNN(sliceResize)
    model_vertebra = cnn_vertebra.loadTestvertebra()

    TruePointNum = 0
    TP = 0 #在目标内且 如果分类正确
    FP = 0 #在目标内且 分类错误 and 如果预测的点不落在任何标注点
    FN = 0 #标注点没有被任何预测点正确命中

    vertebraTP = 0
    vertebraFP = 0
    discTP = 0
    discFP = 0



    mm6c = 0
    mmc = 0
    count = 0
    study = os.listdir(testFile)
    for studyI in study:
        Impath = os.listdir(os.path.join(testFile, studyI))
        for Impathi in Impath:
            try:
                studyUid, seriesUid, instanceUid = dicom_metainfo(os.path.join(os.path.join(testFile, studyI), Impathi),
                                                                  ['0020|000d', '0020|000e', '0008|0018'])
                for studyid in truejsonTarge:
                    if studyUid == studyid["studyUid"] and seriesUid == studyid["data"][0][
                        "seriesUid"] and instanceUid == studyid["data"][0]["instanceUid"]:
                        count += 1
                        image = dicom2array(os.path.join(os.path.join(testFile, studyI), Impathi))

                        truepoints = studyid["data"][0]["annotation"][0]["data"]["point"]
                        image = Image.fromarray(image)
                        image = image.convert("RGB")

                        r_image, points = yolo.detect_image(image)
                        #判断检测点与真实点

                        # mean = np.mean(image)
                        # std = np.mean(np.square(image - mean))
                        # image = (image - mean) / np.sqrt(std)

                        image = np.asarray(image)
                        m, n, _ = image.shape

                        matchedIndex = []
                        for truepoint in truepoints:
                            mmc += 1
                            xt, yt = list(map(int, truepoint["coord"]))
                            tag = truepoint["tag"]
                            if "disc" in tag.keys():
                                ct = tag["disc"]
                            else:
                                ct = tag["vertebra"]
                            for index, metricpoint in enumerate(points):
                                c, x, y = metricpoint

                                mm = 6
                                mm8 = 8
                                if x <= xt + mm and x >= xt - mm and y <= yt + mm and y >= yt - mm:  # 在标注点内
                                    matchedIndex.append(index)
                                    mm6c += 1
                                    TruePointNum += 1

                                    w1, h1 = m / 5, n / 20  # 椎间盘要求更长而不是更高  n控制高度
                                    offset = 5

                                    miny = y - h1
                                    maxy = y + h1
                                    minx = x + offset - w1 / 2
                                    maxx = x + offset + w1 / 2

                                    nimg_x = image[int(miny):int(maxy), int(minx):int(maxx)]
                                    xlen2 = int(w1 / 2)
                                    nimg_x_r = nimg_x[:, int(xlen2):]
                                    # nimg_x_r = np.where(nimg_x_r > 0, nimg_x_r, 0)

                                    nimg_x_r = cv2.resize(nimg_x_r, (sliceResize[0], sliceResize[1]))
                                    X = nimg_x_r[:, :, 1].reshape([1] + sliceResize + [1])

                                    if "disc" in tag.keys():
                                        p = model_disc.predict(X)
                                        pre = np.argmax(p, axis=1)
                                        pc = cnn_disc.disc_label[int(pre)]
                                        if pc == ct:
                                            TP += 1
                                            discTP += 1
                                        else:
                                            FP += 1
                                            discFP += 1
                                    else:
                                        w1, h1 = m / 5, n / 10  # 识别框的宽度和高度 更大
                                        miny = y - h1
                                        maxy = y + h1
                                        minx = x - w1 / 2
                                        maxx = x + w1 / 2

                                        nimg_x = image[int(miny):int(maxy), int(minx):int(maxx)]
                                        nimg_x = cv2.resize(nimg_x, (sliceResize[0], sliceResize[1]))[:, :, 1]

                                        X = nimg_x.reshape([1] + sliceResize + [1])
                                        p = model_vertebra.predict(X)
                                        pre = np.argmax(p, axis=1)
                                        pc = cnn_vertebra.vertebra_label[int(pre)]
                                        if pc == ct:
                                            TP += 1
                                            vertebraTP += 1
                                        else:
                                            FP += 1
                                            vertebraFP += 1
                                    break
                        FP += len(points) - len(matchedIndex)
            except:
                print("文件错误：", os.path.join(os.path.join(dataPath, studyI), Impathi))
    print("总共点数:",mmc)
    print("TP/总定位:",TruePointNum/mmc,"TP/(TP+FP):",TP/(TP+FP+1e-2))
    print("discTP/(TP+FP):",discTP/(discTP+discFP),"vertebraTP/(TP+FP):",vertebraTP/(vertebraTP+vertebraFP))


if __name__ == "__main__":
    # detect_img(YOLO())
    # detect_imgs(YOLO(),testpath=r"val.txt")
    # optimiseTxt(r'resultStep1.txt')
    # cv2Testjpg(txtPath=r'resultStep1_op.txt',testjpg=r"VOCdevkit\Test\JPEGImages")

    seestudy = "study238"

    dataPath = r"H:\dataBase\tianchi_spinal\lumbar_testA50"
    metric_dataTestPath = r"H:\dataBase\tianchi_spinal\lumbar_train51\train"
    jsonPath = r"reslut.json"
    truejsonPath = r'H:\dataBase\tianchi_spinal\lumbar_train51_annotation.json'

    metricDetect(YOLO(),truejsonPath,metric_dataTestPath)
    watchTest(dataPath,r"reslut_.json")
    # watchTest1(dataPath, jsonPath=r"reslut_.json" ,truejsonPath=r"reslut_2.json")