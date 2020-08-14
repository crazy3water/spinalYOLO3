import os
import numpy as np
import re
from readyData import dicom2array,dicom_metainfo
from testcode import TraMapSag,SagKeyPoints

def parse_position(data):
    o2 = []
    for i in data.split("\\"):
        o2.append(float(i))
    return np.array(o2)


def parse_pixelSpacing(data):
    pixelSpa = []
    for i in data.split("\\"):
        pixelSpa.append(float(i))
    return np.array(pixelSpa)

def sortPoint(d):
    points = []
    for target in d[1:]:
        x, y, vd, c, p = target.split(',')
        points.append([int(x),int(y)])
    a = np.array(points)
    a_index = np.argsort(a[:,1])[::-1] #从大到小
    d1 = []
    for index in a_index:
        d1.append(d[1:][index])
    return d1

def step1(dataPath,jsondata):
    '''
    将 图片 与 锚点对应
    :return:
    '''
    image = {}
    target = {}
    count = 0
    study = os.listdir(dataPath)
    for studyI in study:
        Impath = os.listdir(os.path.join(dataPath,studyI))
        print("study:",studyI)
        trueFalg = 0
        describes2pos = {}
        describes2imgPathes = {}
        for Impathi in Impath:
            try:
                studyUid, seriesUid, instanceUid = dicom_metainfo(os.path.join(os.path.join(dataPath,studyI),Impathi),['0020|000d', '0020|000e', '0008|0018'])
                for studyid in jsondata:
                    if studyUid == studyid["studyUid"] and seriesUid == studyid["data"][0]["seriesUid"] and instanceUid == studyid["data"][0]["instanceUid"]:
                        count += 1
                        image[studyI+"_"+Impathi] = dicom2array(os.path.join(os.path.join(dataPath,studyI),Impathi))
                        target[studyI+"_"+Impathi] = studyid["data"][0]["annotation"][0]["data"]["point"]
                        trueFalg = 1
                if trueFalg:
                    for Impathi in Impath:
                        pos = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi), ['0020|0032'])[0]
                        describ = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi), ['0008|103e'])[0]
                        seriesNumber = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi), ['0020|0011'])[0]
                        describ += seriesNumber
                        if re.match(r"(.*)[sS][cC][oO][uU][tT](.*)", describ, 0):  # 不加入
                            continue
                        elif re.match(r"(.*)[sS][aA][gG](.*)", describ, flags=0) or re.match(r"(.*)[sS][aA][gG](.*)",
                                                                                             describ, 0):
                            continue
                        else:
                            if describ not in describes2pos.keys():
                                describes2pos[describ] = [parse_position(pos)[0]]
                                describes2imgPathes[describ] = [Impathi]
                            else:
                                describes2pos[describ].append(parse_position(pos)[0])
                                describes2imgPathes[describ].append(Impathi)
                    describes2posN = {}
                    describes2imgPathesN = {}
                    flagN = False
                    for key, value in describes2pos.items():
                        if re.match(r"(.*)[tT][rR][aA](.*)", key, flags=0):
                            flagN = True
                            describes2posN[key] = value
                            describes2imgPathesN[key] = describes2imgPathes[key]
                    if flagN:
                        describes2pos = describes2posN
                        describes2imgPathes = describes2imgPathesN

                    sagPoints = SagKeyPoints(os.path.join(os.path.join(dataPath, studyI), Impathi))
                    for key, value in describes2pos.items():
                        if np.std(np.array(value)) < 10 and len(value) > 6:
                            target[studyI + "_" + Impathi]
                            print("找到tra! -- ", studyI, key, " -- ", len(value))
                            clusData = []
                            for traPath in describes2imgPathes[key]:
                                traDim = os.path.join(os.path.join(dataPath, studyI), traPath)
                                TraMapSagPoints = TraMapSag(sagPoints, traDim)
                                imgTRA = dicom2array(traDim)

                                x1, x2 = int(TraMapSagPoints[0][0]), int(TraMapSagPoints[1][0])
                                y1, y2 = int(TraMapSagPoints[0][1]), int(TraMapSagPoints[1][1])
                                clusData.append([y1, y2, x1, x2, traDim])


            except:
                print("文件错误：",os.path.join(os.path.join(dataPath,studyI),Impathi))
    np.savez("image.npz", dic=image)
    np.savez("target.npz", dic = target)
    print("一共标记数量为:",count)

if __name__ == "__main__":
    import json
    dataPath = r"H:\dataBase\tianchi_spinal\lumbar_train150"
    jsonPath = r"H:\dataBase\tianchi_spinal\lumbar_train150_annotation.json"
    with open(jsonPath,"r",encoding="utf-8") as f:
        jsonTarge = json.loads(f.read())
    f.close()
    step1(dataPath,jsondata=jsonTarge)