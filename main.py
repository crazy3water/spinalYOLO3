import readyData
import yolo
import regression
import classifycation
import json
import numpy as np
import train
import optimise

Temp = {
    "studyUid": "1.2.840.473.8013.20181026.1091511.864.22434.92",  # dicom study UID
    "data": [
        {
            "instanceUid": "1.3.46.670589.11.32898.5.0.16316.2018102610554807009", # dicom instance UID
            "seriesUid": "1.3.46.670589.11.32898.5.0.16316.2018102610554554000", # dicom series UID
            "annotation": [
                {
                    "annotator": 0  ,          #  可选
                    "point": [                   #关键点标注
                        {
                            "coord": [252, 435],        #点的像素坐标
                            "tag": {
                                "disc": "v2",                 # 椎间盘类型膨出
                                "identification":"L1-L2"      # 椎间盘定位L1-L2间椎间盘
                            },
                            "zIndex": 0,                #第几个slice
                        },
                    ]
                }
            ]
        }
    ]
}

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

def to_Json(pointsPath,dataTestPath,Tojson=r"reslut_.json"):
    jsonFile = []

    identificationList = ["L5-S1" ,"L5" ,"L4-L5","L4" ,"L3-L4" ,"L3"
                          ,"L2-L3","L2","L1-L2" ,"L1" ,"T12-L1" ,"T12"]
    f = open(pointsPath,'r')
    datas = f.readlines()
    count = 0

    for data in datas:
        Temp = {
            "studyUid": "1.2.840.473.8013.20181026.1091511.864.22434.92",  # dicom study UID
            "data": [
                {
                    "instanceUid": "1.3.46.670589.11.32898.5.0.16316.2018102610554807009",  # dicom instance UID
                    "seriesUid": "1.3.46.670589.11.32898.5.0.16316.2018102610554554000",  # dicom series UID
                    "annotation": [
                        {
                            "data":[
                                {
                                    "annotator": 0,  # 可选
                                    "point": [  # 关键点标注
                                        {
                                            "coord": [252, 435],  # 点的像素坐标
                                            "tag": {
                                                "disc": "v2",  # 椎间盘类型膨出
                                                "identification": "L1-L2"  # 椎间盘定位L1-L2间椎间盘
                                            },
                                            "zIndex": 0,  # 第几个slice
                                        },
                                    ]
                                }
                            ],

                        }
                    ]
                }
            ]
        }
        jsonTemp = Temp
        d = data.split()
        ImgPath = d[0].split('/')[-1].split('.')[0].split('_')  #寻找dcm
        dicmPath = dataTestPath+'/'+ImgPath[0]+"/"+ImgPath[1]+'.dcm'
        studyUid, seriesUid, instanceUid,instanceNumber = readyData.dicom_metainfo(dicmPath,['0020|000d', '0020|000e', '0008|0018','0020|0013'])
        jsonTemp["studyUid"] = studyUid
        jsonTemp["data"][0]["instanceUid"] = instanceUid
        jsonTemp["data"][0]["seriesUid"] = seriesUid
        jsonTemp["data"][0]["annotation"] = []
        annotationDic = {}
        annotationDic["data"] = {}
        annotationDic["data"]["point"] = []

        if len(d) > 1:
            d = sortPoint(d)
            for index,target in enumerate(d):
                tagDic = {}
                point_dic = {}
                x,y,vd,c,p = target.split(',')
                # c = "v2"
                point_dic["coord"] = [int(x),int(y)]
                if vd == "disc":
                    if p == '1':
                        tagDic[vd] = c
                    else:
                        tagDic[vd] = c
                else:
                    tagDic[vd] = c
                # tagDic["p"] = p
                tagDic["identification"] = identificationList[index]
                point_dic["tag"] = tagDic
                point_dic["zIndex"] = instanceNumber
                annotationDic["data"]["point"].append(point_dic)
                if index == 11:
                    break

        jsonTemp["data"][0]["annotation"].append(annotationDic)
        jsonFile.append(jsonTemp)
        count += 1
    f.close()

    f1  = open(Tojson,"w")
    json.dump(jsonFile,f1)
    f1.close()

    print("json中数量:",count)

if __name__ == "__main__":
    '''
    在整个项目开始之前到preclassify.py中预训练判断t2 t1
    from preclassify import CNN
    cnn = CNN()
    cnn.train_T()
    '''


    trainFlag = False
    trainAndTest = False

    dataPath = r"H:\dataBase\tianchi_spinal\lumbar_train150"
    jsonPath = r"H:\dataBase\tianchi_spinal\lumbar_train150_annotation.json"

    dataTestPath = r"H:\dataBase\tianchi_spinal\lumbar_testA50"

    metric_dataTestPath = r"H:\dataBase\tianchi_spinal\lumbar_train51\\train"
    metric_jsonPath = r"H:\dataBase\tianchi_spinal\lumbar_train51_annotation.json"

    # yolo.metric_json(r"H:\MyProject\tianchi\spinalYOLO3\reslut_.json",metric_jsonPath)
    # import sys
    # sys.exit()
    if trainFlag:
        # ------------------训练---------------------
            #------------------定位数据准备---------------------
        # readyData.ready(dataPath,jsonPath)
        # train._main()
        #     ------------------分类数据准备---------------------
        # trainClf = r"trainClf1.txt"
        # valClf = r"valClf1.txt"
        # readyData.getTrinClf(trainClf, flag="train",sliceResize=[48, 32])
        # readyData.getTrinClf(valClf, flag="val",sliceResize=[48, 32])
        classifycation._main()
    else:
        #------------------测试---------------------
            #------------------测试数据准备---------------------
        # print("准备：jpg test.txt 在测试准备数据时还存在一个cnn网络分类t1 t2")
        readyData.step1Test(dataPath=dataTestPath,Totxt=r"test.txt")

        # print("检测")
        yolo.detect_imgs(yolo.YOLO(),testpath=r"test.txt")   #得到结果---->resultStep1.txt

        print("优化 resultStep1.txt")
        optimise.clearTxt(r"resultStep1.txt",optxt =r'resultStep_clear.txt',dataTestPath=dataTestPath)
        optimise.train_regression2test(train=False, getResultPath=r"resultStep_clear.txt",
                                         toFile="resultStep_regression.txt")


        optimise.optimiseTxt1(r"resultStep_regression.txt", optxt=r'resultStep_optimise1.txt', dataTestPath=dataTestPath)

        optimise.train_regression2test(train=False, getResultPath=r"resultStep_optimise1.txt",
                                         toFile="resultStep_regression.txt")

        optimise.optimiseTxt2(r"resultStep_regression.txt", optxt=r'resultStep_optimise2.txt',
                              dataTestPath=dataTestPath)

        # optimise.optimiseTxt(r"resultStep_regression.txt",optxt =r'resultStep_optimise.txt',dataTestPath=dataTestPath)

        print("准备切片分类")
        classifycation.ReadySlice2class(dataTxt=r"resultStep_optimise2.txt",resultTxt=r"resultStep3_.txt",sliceResize=[48, 32])                   #得到结果---->resultStep3.txt
        to_Json(r"resultStep3_.txt",dataTestPath,Tojson=r"reslut_.json")