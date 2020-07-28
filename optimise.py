import readyData
import re
import numpy as np
import cv2

"""
-----------------------------------------
优化txt文件：
    1.先清除被识别为空的结果，并记录
    2.有结果的进行选择，选择最长的结果。
-----------------------------------------
"""

def getLightValue(sl):
    jpg = sl[0]
    img = cv2.imread(jpg)
    imgMean,imgstd = cv2.meanStdDev(img)
    return max(imgMean),max(imgstd)

def chooseLimit(k,ki,locNum,dataTestPath,SAGFlag = True):
    km = []
    for kj in k:
        if len(kj) >= locNum or len(kj) >= 9:
            study, imgN = kj[0].split('/')[-1].split('.')[0].split('_')
            dcmPath = dataTestPath + '/' + study + '/' + imgN + '.dcm'
            inde = readyData.dicom_metainfo(dcmPath, ['0020|0013'])[0]
            try:
                a = readyData.dicom_metainfo(dcmPath, ['0008|103e'])[0]
            except:
                a = "t2"
            if SAGFlag:
                if re.match(r"(.*)[tT]2(.*)[sS][aA][gG](.*)", a, 0) or \
                        re.match(r"(.*)[sS][aA][gG](.*)[tT]2(.*)", a, 0):
                    if int(inde) > 2 and int(inde) < 8:
                        km.append(kj)
                        ki.append(getLightValue(kj)[0])
                        # ki.append(inde)
                        # print("均值：", getLightValue(kj)[0], "描述：", str(a), " ", inde)
            else:
                if int(inde) > 3 and int(inde) < 9:
                    if re.match(r"(.*)[sS][cC][oO][uU][tT](.*)", a, 0):
                        continue
                    else:
                        km.append(kj)
                        ki.append(getLightValue(kj)[0])
                    # ki.append(inde)
                    # print("均值：", getLightValue(kj)[0], "描述：", str(a), " ", inde)
    return km,ki

def anchorFlag(sl):
    #判断每一个图像的框y是否长 sl:进来的是每个图的框
    coord = []
    for anchor in sl:
        c,x,y = anchor.split(",")
        coord.append([int(x),int(y)])
    coord.sort(key=lambda x:x[1])

    #框y的相对差
    de = []
    for index in range(0,len(coord)-1):
        de.append(coord[index+1][1] - coord[index][1])

    return np.std(np.array(de))

def parse_position(data):
    o2 = []
    for i in data.split("\\"):
        o2.append(float(i))
    return np.array(o2)

def chooseBestAchorForPos(k,ki,locNum,dataTestPath,SAGFlag = True):
    poses = []
    kjs = []
    kjlens = []
    for kj in k:
        study, imgN = kj[0].split('/')[-1].split('.')[0].split('_')
        dcmPath = dataTestPath + '/' + study + '/' + imgN + '.dcm'
        try:
            a = readyData.dicom_metainfo(dcmPath, ['0008|103e'])[0]
        except:
            a = "t2"
        if re.match(r"(.*)[sS][cC][oO][uU][tT](.*)", a, 0):
            continue
        else:
            pos = readyData.dicom_metainfo(dcmPath, ['0020|0032'])[0]
            pos = parse_position(pos)[0]
            poses.append(pos)
            kjs.append(kj)
            kjlens.append(len(kj))
    km = np.array(kjs)[np.array(poses).argsort()[int(len(poses)/2)-1:int(len(poses)/2)+1]]
    kjlens = np.array(kjlens)[np.array(poses).argsort()[int(len(poses)/2)-1:int(len(poses)/2)+1]]
    km = [km[np.array(kjlens).argsort()[-1]]]
    ki = [0]
    return km,ki

def chooseBestAchor(k,ki,locNum,dataTestPath,SAGFlag = True):
    km = []

    for kj in k:
        if len(kj) >= locNum or len(kj) >= 9:
            study, imgN = kj[0].split('/')[-1].split('.')[0].split('_')
            dcmPath = dataTestPath + '/' + study + '/' + imgN + '.dcm'
            inde = readyData.dicom_metainfo(dcmPath, ['0020|0013'])[0]

            pos = readyData.dicom_metainfo(dcmPath, ['0020|0032'])[0]
            pos = parse_position(pos)[0]
            try:
                a = readyData.dicom_metainfo(dcmPath, ['0008|103e'])[0]
            except:
                a = "t2"
            if SAGFlag:
                if int(inde) > 3 and int(inde) < 9:
                    km.append(kj)
                    ki.append(anchorFlag(kj[1:]))
            else:
                if int(inde) > 3 and int(inde) < 9:
                    if re.match(r"(.*)[sS][cC][oO][uU][tT](.*)", a, 0):
                        continue
                    else:
                        km.append(kj)
                        ki.append(getLightValue(kj)[1])
    return km,ki


def clearTxt(resultStep1,optxt =r'resultStep1_op.txt',dataTestPath=None):
    fdata = open(resultStep1, "r")
    fop = open(optxt, 'w')
    data = fdata.readlines()
    for line in data:
        line = line.strip()
        sl = line.split()
        studyName = sl[0].split('_')[0]
        lenloc = len(sl)
        if lenloc > 3:
            fop.write(line+'\n')
    fdata.close()
    fop.close()

def optimiseTxt1(resultStep1,optxt =r'resultStep1_op.txt',dataTestPath=None):
    """
    先选择一些预测点较多的进行保存
    :param resultStep1:
    :param optxt:
    :param dataTestPath:
    :return:
    """
    print("可以得到一堆满足条件的框")
    fdata = open(resultStep1, "r")
    fop = open(optxt, 'w')
    data = fdata.readlines()
    newName = ""
    locNum = 0
    saveline = ""
    k = []
    ki = []
    maxlight = ""
    for line in data:
        line = line.strip()
        sl = line.split()
        studyName = sl[0].split('_')[0]
        lenloc = len(sl)

        if studyName != newName:
            print("Name:", newName)
            if len(k) > 0:

                if len(ki) == 0:
                    km, ki = chooseLimit(k, ki, locNum - 1, dataTestPath, SAGFlag=True)

                if len(ki) == 0:
                    print("第二长定位没有t2tag,采用最长定位中位数")
                    km, ki = chooseLimit(k, ki, 5, dataTestPath, SAGFlag=False)

                for chooseLine in km:
                    saveline = ' '.join(chooseLine)
                    if saveline != "":
                        fop.write(saveline + '\n')
            newName = studyName
            locNum = 0
            saveline = line
            k = [sl]
            ki = []
        else:
            k.append(sl)
            if lenloc > locNum:
                locNum = lenloc
                saveline = line

    if len(k) > 0:
        km, ki = chooseLimit(k, ki, locNum - 1, dataTestPath, SAGFlag=True)
        if len(ki) == 0:
            print("第二长定位没有t2tag,采用最长定位中位数")
            km, ki = chooseLimit(k, ki, 5, dataTestPath, SAGFlag=False)
        for chooseLine in km:
            saveline = ' '.join(chooseLine)
            if saveline != "":
                fop.write(saveline + '\n')
    fdata.close()
    fop.close()

def optimiseTxt2(resultStep1,optxt =r'resultStep1_op.txt',dataTestPath=None):
    """
        选择每个study的框，其中选择的框的条件：
            1.y轴要长
    """
    print("-再选择合适的框")

    fdata = open(resultStep1, "r")
    fop = open(optxt, 'w')
    data = fdata.readlines()
    newName = ""
    locNum = 0
    saveline = ""
    k = []
    ki = []
    maxlight = ""
    for line in data:
        line = line.strip()
        sl = line.split()
        studyName = sl[0].split('_')[0]
        lenloc = len(sl)

        if studyName != newName:
            if len(k) > 0:
                if len(ki) == 0:
                    # km, ki = chooseBestAchor(k, ki, locNum - 1, dataTestPath, SAGFlag=True)
                    km, ki = chooseBestAchorForPos(k, ki, locNum - 1, dataTestPath, SAGFlag=True)
                if len(ki) == 0:
                    print("--第二长定位没有t2tag,采用最长定位中位数")
                    # km, ki = chooseBestAchor(k, ki, 5, dataTestPath, SAGFlag=False)
                    km, ki = chooseBestAchorForPos(k, ki, 5, dataTestPath, SAGFlag=False)

                isort = np.array(ki).reshape([-1]).argsort()
                k = np.array(km)[isort]
                saveline = ' '.join(k[0])
            if saveline != "":
                fop.write(saveline + '\n')
            newName = studyName
            locNum = 0
            saveline = line
            k = [sl]
            ki = []
        else:
            k.append(sl)
            if lenloc > locNum:
                locNum = lenloc
                saveline = line

    if len(k) > 0:
        km, ki = chooseBestAchor(k, ki, locNum - 1, dataTestPath, SAGFlag=True)
        if len(ki) == 0:
            print("--第二长定位没有t2tag,采用最长定位中位数")
            km, ki = chooseBestAchor(k, ki, 5, dataTestPath, SAGFlag=False)
        isort = np.array(ki).reshape([-1]).argsort()
        k = np.array(km)[isort]
        saveline = ' '.join(k[0])

    fop.write(saveline)  # 保存最后一个

    fdata.close()
    fop.close()

def optimiseTxt(resultStep1,optxt =r'resultStep1_op.txt',dataTestPath=None):
    """
    选择每个study的框，其中选择的框的条件：
        1.y轴要长
    """
    fdata = open(resultStep1,"r")
    fop = open(optxt, 'w')
    data = fdata.readlines()
    newName = ""
    locNum = 0
    saveline = ""
    k = []
    ki = []
    maxlight = ""
    for line in data:
        line = line.strip()
        sl = line.split()
        studyName = sl[0].split('_')[0]
        lenloc = len(sl)

        if studyName != newName:
            print("Name:",newName)
            if len(k) > 0:

                if len(ki) == 0:
                    km,ki = chooseLimit(k,ki,locNum-1,dataTestPath,SAGFlag=True)

                if len(ki) == 0:
                    print("第二长定位没有t2tag,采用最长定位中位数")
                    km, ki = chooseLimit(k, ki, 5, dataTestPath, SAGFlag=False)

                isort = np.array(ki).reshape([-1]).argsort()
                k = np.array(km)[isort]
                index = int(len(k)/2)
                saveline = ' '.join(k[index])
            if saveline != "":
                fop.write(saveline+'\n')
            newName = studyName
            locNum = 0
            saveline = line
            k= [sl]
            ki = []
        else:
            k.append(sl)
            if lenloc > locNum:
                locNum = lenloc
                saveline = line

    if len(k)>0:
        km, ki = chooseLimit(k, ki, locNum - 1, dataTestPath, SAGFlag=True)
        if len(ki) == 0:
            print("第二长定位没有t2tag,采用最长定位中位数")
            km, ki = chooseLimit(k, ki, 5, dataTestPath, SAGFlag=False)
        isort = np.array(ki).reshape([-1]).argsort()
        k = np.array(km)[isort]
        index = int(len(k)/2)
        saveline = ' '.join(k[index])

    fop.write(saveline) # 保存最后一个


    fdata.close()
    fop.close()

"""
-----------------------------------------
优化txt文件：
    1.坐标异常点清除
    2.最下方左边回归
-----------------------------------------
"""

def imgScalePoint(imgPath, coord, resizeShape=None):
    #将图片进行放缩后返回对应的点坐标，从上往下
    if resizeShape is None:
        resizeShape = [256, 256]

    img = cv2.imread(imgPath)
    m, n = img.shape[0], img.shape[1]

    # print("图像大小:", img.shape)

    img = cv2.resize(img, dsize=(resizeShape[0], resizeShape[1]))

    scalexRate = (1.0 * resizeShape[0]) / m
    scaleyRate = (1.0 * resizeShape[1]) / n
    # print("放缩比例:", scalexRate, scaleyRate)

    coord = np.array(coord)
    minIndex = np.argsort(coord[:, 1])

    coord = coord[minIndex]

    img = cv2.circle(img, (int(coord[-1][0] * scaleyRate), int(coord[-1][1] * scalexRate)), radius=3, color=[0, 0, 255],
                     thickness=1)
    # print("最低位x:{},y:{}".format(coord[-1][0] * scaleyRate, coord[-1][1] * scalexRate))
    for x, y in coord:
        img = cv2.circle(img, (int(x * scaleyRate), int(y * scalexRate)), radius=1, color=[0, 0, 255], thickness=-1)

    # cv2.imshow(' ', img)
    # cv2.waitKey(0)
    return coord

def getResult(filepath):
    f = open(filepath,'r')
    datas = f.readlines()
    coords = []
    tags = []
    images = []
    for data in datas:  #每一行数据
        images.append(data.split()[0])  # jpg图片路径
        coord = []
        tag = []
        for d in data.split()[1:]:
            tag.append(d.split(",")[0]) #保存每一个框 椎体类型
            coord.append(list(map(int,d.split(",")[1:])))
        tags.append(tag)
        coords.append(coord)
    f.close()
    return coords,tags,images

def getData(path):
    fTrain = open(path,"r")
    datas = fTrain.readlines()
    train = []
    test = []
    for data in datas:
        data = data.split()
        imgP = data[0]
        coords = []
        for point in data[1:]:
            minx,miny,maxx,maxy = list(map(float,point.split(',')))[:-1]
            centerX,centerY = int(minx + (maxx-minx)/2),int(miny + (maxy-miny)/2)
            coords.append([centerX,centerY])
        dataX = imgScalePoint(imgP,coords,resizeShape=[256,256])
        #将dataX打开，重新构造特征满足 x = f(i,y) --> 预测x：（0,y+dy）= ?
        for index,value in enumerate(dataX):
            x,y = value
            if index == 10:
                test.append([1,y,x])  # 取末尾
            train.append([11 - index,y,x])  #取末尾之前

    train = np.array(train)
    test = np.array(test)
    print("数据集:",train.shape,test.shape)
    fTrain.close()
    return train,test

def optimiseResult(coords,tags,images,model=None):
    #  对结果进行排序 y从大到小 依次为从下到上
    ncoords,ntags = [],[]
    for coord,tag,img in zip(coords,tags,images):


        coord = np.array(coord)
        minIndex = np.argsort(coord[:,1])

        coord = coord[minIndex]
        tag = np.array(tag)[minIndex]
        #-------------------可视化----------------------------
        # img = cv2.circle(img,(int(coord[-1][0]*scaleyRate),int(coord[-1][1]*scalexRate)),radius=3,color=[0,0,255],thickness=1)
        # print("最低位x:{},y:{},类型：{}".format(coord[-1][0] * scaleyRate, coord[-1][1] * scalexRate, tag[-1]))
        # for x,y in coord:
        #     img = cv2.circle(img, (int(x*scaleyRate),int(y*scalexRate)), radius=1, color=[0, 0, 255], thickness=-1)
        #     print("x,y:",x*scaleyRate,y*scalexRate)

        dy = 0
        dx = 0
        if tag[-1] == "vertebra": #说明最下面是锥柱 需要拟合优化
            print(img,"--生成最下面的点")
            #先评估y的大概位置，再使用clf预测x
            lowx = int(coord[-1][0])
            lowy = int(coord[-1][1])
            for i in range(len(coord)-1):
                dy += abs(coord[i+1][1] - coord[i][1])
                dx += abs(coord[i + 1][0] - coord[i][0])
            dy = int(dy / (len(coord) - 1))
            dx = int(dx / (len(coord) - 1))

            nextx = lowx+5

            nextpoint = np.array([int(nextx),int(lowy+dy)]).reshape([1,-1])
            coord = np.concatenate((coord,nextpoint),axis=0)
            nexttag = np.array(["disc"])
            tag = np.concatenate((tag,nexttag),axis=0)
        ncoords.append(coord), ntags.append(tag)

    return ncoords,ntags,images

def saveStep2(ncoords,ntags,imgs,toFile="resultStep2.txt"):
    f = open(toFile,'w')
    for ncoord,ntag,img in zip(ncoords,ntags,imgs):
        f.write(img)
        for coord,tag in zip(ncoord,ntag):
            f.write(" "+tag+','+str(coord[0])+','+str(coord[1]))
        f.write("\n")
    f.close()

def computDis(coord,targetX):
    sum = 0
    for i in range(len(coord)):
        x, y = coord[i][0], coord[i][1]
        sum += abs(x-targetX)
    return sum

def delAway(coords,tags,images):
    #删除 偏离的 点
    #重复 在6个像素点内
    ncoords, ntags,imgs = [], [], []
    for coord,tag,img in zip(coords,tags,images):
        if len(coord) > 5:
            #将坐标点y排序
            coord = np.array(coord)
            minIndex = np.argsort(coord[:,1])

            coord = coord[minIndex]
            tag = np.array(tag)[minIndex]

            recordy = []
            for i in range(len(coord)-1):
                recordy.append(abs(coord[i+1][1] - coord[i][1]))
            recordy = np.array(recordy).reshape([-1])
            zwy = np.median(recordy)

            study, imgN = img.split('/')[-1].split('.')[0].split('_')
            if study == "study245" and imgN == "image17":
                print()

            delIndex = 1
            while(delIndex):
                for i in range(len(coord)-1):
                    if tag[i+1] == tag[i] and \
                    abs(coord[i+1][1] - coord[i][1]) < zwy and abs(coord[i + 1][0] - coord[i][0]) < zwy:#zwy*0.5:
                        coord[i][1],coord[i][0] = int((coord[i+1][1] + coord[i][1]) /2),int((coord[i+1][0] + coord[i][0]) /2)
                        coord = np.delete(coord,i+1,axis=0)
                        tag = np.delete(tag,i+1,axis=0)
                        print("删除{}重合点".format(img))
                        delIndex = 1
                        break
                    delIndex =0



            # d1 = throldCluster(coord[:,0])
            # coord = np.delete(coord, d1, axis=0)
            # tag = np.delete(tag, d1, axis=0)
            # ------------------x------------------
            # record = []
            # data = coord[:,0]
            # for i in range(len(data)):
            #     record.append(computDis(coord,data[i]))
            #
            # delflag = 1
            #
            # dis = record
            # d1 = []
            # i = len(dis)-1
            # while(i>=0):
            #     targetDis = dis[i]
            #     if targetDis > 2*np.mean(dis[:i]+dis[i:]):
            #         print("{}第{}个点x偏离".format(img,i))
            #         d1.append(i)
            #         print(data)
            #         print(np.percentile(data, 0.8))
            #         i -= 2
            #         # break
            #     else:
            #         delflag = 0
            #         i -= 1
            #
            # coord = np.delete(coord, d1, axis=0)
            # tag = np.delete(tag, d1, axis=0)
            # #------------------y------------------
            # record = []
            # data = coord[:,1]
            # for i in range(len(data)):
            #     record.append(computDis(coord,data[i]))
            #
            # dis = record
            # d1 = []
            # i = len(dis)-1
            # while(i>=0):
            #     targetDis = dis[i]
            #     if targetDis > 2.0*np.mean(dis[:i]+dis[i:]):
            #         print("{}第{}个点y偏离".format(img,i))
            #         d1.append(i)
            #         print(data)
            #         print(np.percentile(data, 0.8))
            #         i -= 2
            #         # break
            #     else:
            #         i -= 1
            #
            # coord = np.delete(coord, d1, axis=0)
            # tag = np.delete(tag, d1, axis=0)
            # ------------------y------------------
            ncoords.append(coord)
            ntags.append(tag)
            imgs.append(img)
    return ncoords,ntags,imgs

def train_regression2test(getResultPath,train=None,toFile="resultStep2.txt"):
    clf = None

    ncoords,ntags,imgs = getResult(getResultPath)

    print("对检测结果回归")
    #删除8mm点
    ncoords,ntags,imgs = delAway(ncoords,ntags,imgs)
    # 缺失点
    # ncoords, ntags, imgs = midPoint(ncoords, ntags, imgs)
    # 尾部回归点
    # ncoords,ntags,imgs = optimiseResult(ncoords,ntags,imgs,model=clf)

    saveStep2(ncoords,ntags,imgs,toFile)
