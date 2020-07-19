import numpy as np
import readyData

def kmeans(studyPoint,throld=30):
    xIndex = np.argsort(np.array(studyPoint[:,1],np.float))
    studyPoint = studyPoint[xIndex]
    prex = 0
    blockIndex = 0
    blockConllection = {}
    for point in studyPoint:
        c, x, y, index, imagePath = point
        x = int(x)
        if abs(x - prex) >= throld:  # 超出范围，新的块
            blocky = [point]
            blockIndex += 1
            blockConllection[blockIndex] = blocky
        else:
            blockConllection[blockIndex].append(point)
        prex = x

    if len(blockConllection.keys()) == 2:
        print("-----------------------异常点X-----------------")
        points = list(blockConllection.values())
        if len(points[0]) > len(points[1]) :
            print("出现异常点：{}".format((points[0][1][1],points[0][1][2])))
        else:
            print("出现异常点：{}".format((points[0][0][1], points[0][0][2])))



def getNewPoints(throld,studyPoint,prestudyName):
    NewPoints = []
    # 1.按y排序
    studyPoint = np.array(studyPoint)[studyPoint[:,1]!=str(0.0)]

    kmeans(studyPoint)

    yIndex = np.argsort(np.array(studyPoint[:,2],np.float))
    studyPoint = studyPoint[yIndex]
    #从最小的开始遍历，将10mm内的进行分块处理
    prey = 0
    blockIndex = 1
    blockConllection = {}
    for point in studyPoint:
        c, x,y,index,imagePath = point
        y = int(y)
        if abs(y-prey) >= throld: #超出范围，新的块
            blockIndex += 1
            blocky = [point]
            blockConllection[blockIndex] = blocky
        else:
            blockConllection[blockIndex].append(point)
        prey = y

    for key,value in blockConllection.items():
        if len(value) == 2:
            print("-----------------------异常点-----------------")
            print("{}出现异常点：块：{}，点：{}".format(prestudyName,key,value))

    # print("-----------{}结果点-----------------".format(prestudyName))
    # for key,value in blockConllection.items():
    #     print("出现点：块：{}，点：{}".format(key,value))

    #2.取点 ： 中间的数
    for key,value in blockConllection.items():
        vlen = len(value)
        if vlen > 1:
            NewPoints.append(value[int(vlen/2)])


    # zIndex = [0]*10
    # zIndex2ImagPath = {}
    # for point in NewPoints:
    #     c, x, y, index, imagePath = point
    #     zIndex[int(index)] += 1
    #     if index not in zIndex2ImagPath.keys():
    #         zIndex2ImagPath[int(index)] = imagePath

    for key,value in blockConllection.items():
        vlen = len(value[0])
        if vlen > 1:
            NewPoints.append(value[0])
    return NewPoints,NewPoints[0][-1]

import re
def matchSAG(a,inde):
    if re.match(r"(.*)[tT]2(.*)[sS][aA][gG](.*)", a, 0) or \
            re.match(r"(.*)[sS][aA][gG](.*)[tT]2(.*)", a, 0):
        if int(inde) > 2 and int(inde) < 8:
            return True
    return False
def matchIndex(a,inde):
    if int(inde) > 2 and int(inde) < 8:
        return True
    return False

if __name__ == "__main__":
    dataTestPath = r"H:\dataBase\tianchi_spinal\lumbar_testA50"
    optxt = r'resultStep1_op_test.txt'

    fop = open(optxt, 'w')
    fdata = open(r"resultStep1_op.txt","r")

    data = fdata.readlines()

    prestudyName = ""
    prestudydcmPath = ""
    presl = ""
    studyPoint = np.zeros([1,5])
    for line in data:
        line = line.strip()
        sl = line.split()

        if len(sl[1:])>0:
            point = []
            for i in sl[1:]:
                study, imgN = sl[0].split('/')[-1].split('.')[0].split('_')
                dcmPath = dataTestPath + '/' + study + '/' + imgN + '.dcm'
                imagePath = dataTestPath + '/' + study + '_' + imgN + '.jpg'
                index = readyData.dicom_metainfo(dcmPath, ['0020|0013'])[0]
                try:
                    a = readyData.dicom_metainfo(dcmPath, ['0008|103e'])[0]
                except:
                    a = "t2"
                if matchSAG(a,index):
                    # x,y,index
                    point.append([i.split(',')[0],int(i.split(',')[1]),int(i.split(',')[2]),int(index),sl[0]])
                elif matchIndex(a,index):
                    point.append([i.split(',')[0], int(i.split(',')[1]), int(i.split(',')[2]), int(index), sl[0]])
            point = np.array(point)
        else:
            point = np.zeros([1, 5])

        studyName = sl[0].split('_')[0]

        if prestudyName == studyName:
            studyPoint = np.concatenate([studyPoint,point])
            # studyPoint.append(point)
        else:
            if len(studyPoint) > 1 :
                #将同一个study的点组合成新的东西
                ite = 0
                thold = 10
                NewPoints,imagePathBest = getNewPoints(thold, studyPoint, prestudyName)
                lenHis = [len(NewPoints)]
                while (len(NewPoints) < 11):
                    thold -= 1
                    NewPoints,imagePathBest = getNewPoints(thold, studyPoint, prestudyName)
                    ite += 1
                    print("Iter:",ite)
                    lenHis.append(len(NewPoints))
                    if ite > 4:
                        thold =10 - (np.argmax(np.array(lenHis)))* 1
                        NewPoints,imagePathBest = getNewPoints(thold, studyPoint, prestudyName)
                        break
                print("-----------{}结果点--长度{}-----------------".format(prestudyName, len(NewPoints)))
                for i in NewPoints:
                    print(i)
            #找到点，写入
                saveline = imagePathBest
                for pointi in NewPoints:
                    c,x,y,z,imagePath = pointi
                    saveline +=" " + ",".join([c,x,y])
                fop.write(saveline+"\n")
            studyPoint = point
            # studyPoint = []
            # studyPoint.append(point)
            prestudyName = studyName
            prestudydcmPath = dcmPath
            presl = sl[0]

    ite = 0
    thold = 10
    NewPoints,imagePathBest = getNewPoints(thold, studyPoint, prestudyName)
    lenHis = [len(NewPoints)]
    while (len(NewPoints) < 11):
        thold -=  1
        NewPoints,imagePathBest = getNewPoints(thold, studyPoint, prestudyName)
        ite += 1
        print("Iter:", ite)
        lenHis.append(len(NewPoints))
        if ite > 4:
            thold = 10 - np.argmax(np.array(lenHis)) * 1
            break
    print("-----------{}结果点--长度{}-----------------".format(prestudyName, len(NewPoints)))
    for i in NewPoints:
        print(i)
    # 找到点，写入
    saveline = sl[0]
    for pointi in NewPoints:
        c, x, y, z, imagePath = pointi
        saveline += " " + ",".join([c, x, y])
    fop.write(saveline + "\n")
    fop.close()