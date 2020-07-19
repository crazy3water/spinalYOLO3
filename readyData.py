import SimpleITK as sitk
import cv2
import os
import sys
import json
import numpy as np
import re
from preclassify import CNN
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def dicom_metainfo(dicm_path, list_tag):
    '''
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    '''
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    return [reader.GetMetaData(t) for t in list_tag]


def dicom2array(dcm_path):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
            image = sitk.Cast(image, sitk.sitkUInt8)
    img_x = sitk.GetArrayFromImage(image)[0]
    return img_x

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
        for Impathi in Impath:
            try:
                studyUid, seriesUid, instanceUid = dicom_metainfo(os.path.join(os.path.join(dataPath,studyI),Impathi),['0020|000d', '0020|000e', '0008|0018'])
                for studyid in jsondata:
                    if studyUid == studyid["studyUid"] and seriesUid == studyid["data"][0]["seriesUid"] and instanceUid == studyid["data"][0]["instanceUid"]:
                        count += 1
                        image[studyI+"_"+Impathi] = dicom2array(os.path.join(os.path.join(dataPath,studyI),Impathi))
                        target[studyI+"_"+Impathi] = studyid["data"][0]["annotation"][0]["data"]["point"]
            except:
                print("文件错误：",os.path.join(os.path.join(dataPath,studyI),Impathi))
    np.savez("image.npz", dic=image)
    np.savez("target.npz", dic = target)
    print("一共标记数量为:",count)

def img2jpg():
    #读取保存图片的名字，然后返回所有图片的名字
    imgPath = r"VOCdevkit\VOC2007\JPEGImages"
    images = np.load(r"image.npz", allow_pickle=True)["dic"][()]
    target = np.load(r"target.npz", allow_pickle=True)["dic"][()]
    jpgs = []
    print(len(images))
    for imageId in images.keys():
        img_x = images[imageId]
        m,n = img_x.shape
        # print(m, n)
        imgP = imgPath+'/'+imageId[:-4]+'.jpg'
        # jpgs.append(imageId[:-4]+'.jpg')   #保存jpg名字
        jpgs.append(imgP)  # 保存jpg地址
        # f.write(imgP[3:]+"\n")
        cv2.imwrite(imgP,img_x)  #保存jpg文件

    # 分训练集和验证集
    np.random.seed(2020)
    all_ = np.arange(0, len(jpgs))
    valIndex = np.random.randint(low=0, high=len(jpgs), size=int(len(jpgs) * 0.2))
    trainIndex = [i for i in all_ if i not in valIndex]
    f = open(r"train.txt", 'w')
    fc = open(r"trainClf1.txt",'w')
    for train in trainIndex:
        imageName = jpgs[train].split('/')[-1][:-4] + '.dcm'
        f.write(jpgs[train])
        fc.write(jpgs[train])
        xwhconvert(images, target, imageName, f, fc)
        f.write("\n")
        fc.write("\n")
    f.close()
    fc.close()
    #对训练集和验证的锥柱和椎间盘做切片用于训练
    f = open(r"val.txt", 'w')
    fc = open(r"valClf1.txt", 'w')
    for val in valIndex:
        imageName = jpgs[val].split('/')[-1][:-4]+ '.dcm'
        f.write(jpgs[val] )
        fc.write(jpgs[val])
        xwhconvert(images, target, imageName, f, fc)
        f.write("\n")
        fc.write("\n")
    f.close()
    fc.close()
    # np.save(r"data2class/imgesSlice_vertebra_val.npy",np.array(imgesSlice_vertebra_val))
    # np.save(r"data2class/ySlice_vertebra_val.npy", np.array(ySlice_vertebra_val))
    # np.save(r"data2class/imgesSlice_disc_val.npy", np.array(imgesSlice_disc_val))
    # np.save(r"data2class/ySlice_disc_val.npy", np.array(ySlice_disc_val))

def dicom2array1(dcm_path,cnnModel=None):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    flag = True
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        inde = int(image_file_reader.GetMetaData('0020|0013'))
        try:
            description = image_file_reader.GetMetaData('0008|103e')
        except:
            description = "T1"
        if inde >=2 and inde <=7: #只取 zindex :2-6
            if re.match(r"(.*)[tT]2(.*)[sS][aA][gG]",description,flags=0) or re.match(r"(.*)[sS][aA][gG](.*)[tT]2(.*)",description,0):

                if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
                    image = sitk.InvertIntensity(image, maximum=255)
                image = sitk.Cast(image, sitk.sitkUInt8)
            elif  re.match(r"(.*)[tT]2(.*)",description,0):
                if inde >=4 and inde <=8:
                    if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
                        image = sitk.InvertIntensity(image, maximum=255)
                    image = sitk.Cast(image, sitk.sitkUInt8)
            else:
                if re.match(r"(.*)[sS][cC][oO][uU][tT](.*)", description, 0):
                    flag = False
                elif image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
                    image = sitk.InvertIntensity(image, maximum=255)
                image = sitk.Cast(image, sitk.sitkUInt8)
                img_x = sitk.GetArrayFromImage(image)[0]
                img = cv2.resize(img_x,(256,256))
                pre = cnnModel.test_(img)[0]
                if np.argmax(pre) != 0:
                    flag = False
        else:
            flag = False
    if flag:
        img_x = sitk.GetArrayFromImage(image)[0]
    else:
        img_x = "None"
    return img_x

def totarget(datas,img_x,ratex,ratey):
    #获得最左上，最右下
    m, n, c = img_x.shape

    mLeftup,mRightlow = [n,0],[0,m]

    for data in datas:
        x, y = data["coord"]

        tm = 17
        tn = 20

        if x + int(m / tm)>mRightlow[0] :
            mRightlow[0] = x + int(m / tm)
        if y - int(n / tn) < mRightlow[1]:
            mRightlow[1] =  y - int(n / tn)

        if x - int(m / tm)<mLeftup[0]:
            mLeftup[0] = x - int(m / tm)
        if y + int(n / tn) > mLeftup[1]:
            mLeftup[1] = y + int(n / tn)

        identification = data["tag"]["identification"]
        img_x_ = img_x.copy()
        # print(data["tag"])
        # if len(identification) < 3:
        #     # if "vertebra" in data["tag"].keys():
        #     kind = data["tag"]["vertebra"]
        #     # cv2.circle(img_x, (x, y), radius=3, color=[255, 0, 0], thickness=1)
        #     cv2.rectangle(img_x,pt1=(x-int(m/tm),y+int(n/tn)) , pt2=((x+int(m/tm),y-int(n/tn))),
        #                   color=[255,0,0],thickness=-1)
        #     cv2.putText(img_x, kind, (x +3, y), font, 0.3, (255, 255, 255), 1)
        # else:
        #     kind = data["tag"]["disc"]
        #     # cv2.circle(img_x, (x, y), radius=3, color=[255, 0, 0], thickness=-1)
        #     cv2.rectangle(img_x, pt1=(x - int(m / tm), y + int(n / tn)), pt2=((x + int(m / tm), y - int(n / tn))),
        #                   color=[255, 0, 0], thickness=1)
        #     cv2.putText(img_x, kind, (x +3, y), font, 0.3, (255, 255, 255), 1)

    cv2.rectangle(img_x, pt1=(mLeftup[0],mLeftup[1]), pt2=(mRightlow[0],mRightlow[1]),
                  color=[255, 0, 0], thickness=-1)

    img_b = np.zeros([m, n, c],dtype=np.uint8)
    img_b[mRightlow[1]:mLeftup[1],mLeftup[0]:mRightlow[0],:] = img_x_[mRightlow[1]:mLeftup[1],mLeftup[0]:mRightlow[0],:]
    cv2.imshow("img_b", img_b)

    img_x = img_b.copy()
    for data in datas:
        x, y = data["coord"]

        tm = 25
        tn = 25
        identification = data["tag"]["identification"]
        # print(data["tag"])
        if len(identification) < 3:
            # if "vertebra" in data["tag"].keys():
            kind = data["tag"]["vertebra"]
            # cv2.circle(img_x, (x, y), radius=3, color=[255, 0, 0], thickness=1)
            cv2.rectangle(img_x,pt1=(x-int(m/tm),y+int(n/tn)) , pt2=((x+int(m/tm),y-int(n/tn))),
                          color=[255,0,0],thickness=1)
            cv2.putText(img_x, kind, (x +3, y), font, 0.3, (255, 255, 255), 1)
        else:
            kind = data["tag"]["disc"]
            # cv2.circle(img_x, (x, y), radius=3, color=[255, 0, 0], thickness=-1)
            cv2.rectangle(img_x, pt1=(x - int(m / tm), y + int(n / tn)), pt2=((x + int(m / tm), y - int(n / tn))),
                          color=[255, 0, 0], thickness=1)
            cv2.putText(img_x, kind, (x +3, y), font, 0.3, (255, 255, 255), 1)
    cv2.imshow("1", img_x)

    return img_x

className = ["Normal","Degeneration","Bulge","Protruded","Extruded", "Schmor"]

def saveclassName():
    f = open(r"class.names",'w')
    for name in className:
        f.write(name+"\n")
    f.close()


def step1Test(dataPath,Totxt = r"test.txt"):
    '''
    在准备jpg时，首先先判断是否为T2_sag，只对T2_sag做定位识别
    target: t2sag:0, t1sag:1, t2tra:2
    :param dataPath:
    :return:
    '''
    image = {}
    count = 0
    imgPath = r"VOCdevkit\Test\JPEGImages"
    jpgs = []
    study = os.listdir(dataPath)
    cnn = CNN()
    cnn.loadModel() #logs/003/xx.h5

    for studyI in study:
        Impath = os.listdir(os.path.join(dataPath, studyI))
        print("study:", studyI)
        for Impathi in Impath:
            studyUid, seriesUid, instanceUid = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi),
                                                              ['0020|000d', '0020|000e', '0008|0018'])
            img = dicom2array1(
                os.path.join(os.path.join(dataPath, studyI), Impathi),cnnModel=cnn)

            if img != "None":
                # image[studyI + "_" + Impathi] = img
                imgP = imgPath + '/' + studyI + "_" + Impathi[:-4] + '.jpg'
                jpgs.append(imgP)  # 保存jpg地址
                cv2.imwrite(imgP, img)
                count += 1
                # cv2.imshow(" ",img)
                # cv2.waitKey(0)
                # break
    # np.savez("image.npz", dic=image)
    print("In step 1,the number is:", count)
    #制作 test.txt
    f = open(Totxt, 'w')
    for i in range(len(jpgs)):
        f.write(jpgs[i])
        f.write("\n")
    f.close()

    del cnn
    pass

def img2jpgTest():
    #读取保存图片的名字，然后返回所有图片的名字
    imgPath = r"VOCdevkit\Test\JPEGImages"
    images = np.load(r"image.npz", allow_pickle=True)["dic"][()]
    jpgs = []
    print("The number of test:",len(images))
    for imageId in images.keys():
        img_x = images[imageId]
        m,n = img_x.shape
        # print(m, n)
        imgP = imgPath+'/'+imageId[:-4]+'.jpg'
        # jpgs.append(imageId[:-4]+'.jpg')   #保存jpg名字
        jpgs.append(imgP)  # 保存jpg地址
        # f.write(imgP[3:]+"\n")
        cv2.imwrite(imgP,img_x)  #保存jpg文件


    #制作 test.txt
    f = open(r"test.txt", 'w')
    for i in range(len(jpgs)):
        f.write(jpgs[i])
        f.write("\n")
    f.close()

def ReadySlice2class():
    f = open(r"resultStep2.txt")
    datas = f.readlines()

    vertebraSlice = {}
    discSlice = {}

    sliceResize = [52, 52]
    for data in datas:
        d = data.split()
        imgPath = d[0]
        img = cv2.imread(imgPath)
        m,n,_ = img.shape
        vertebraSlice[imgPath] = []
        discSlice[imgPath] = []
        for target in d[1:]:
            target = target.split(',')
            x,y = list(map(int,target[1:]))
            # cv2.imshow(" ",nimg_x)
            # cv2.waitKey(0)
            if target[0] == 'disc':
                w1, h1 = m / 6, n / 20  # 识别框的宽度和高度 更大
                offset = 5
                nimg_x = img[int(y - h1 / 2):int(y + h1 / 2), int(x+offset - w1 / 2):int(x+offset + w1 / 2)]
                nimg_x = cv2.resize(nimg_x, (sliceResize[0], sliceResize[1])).reshape(sliceResize + [1])
                # cv2.imshow(" ",nimg_x)
                discSlice[imgPath].append(nimg_x)
            else:
                w1, h1 = m / 6, n / 12  # 识别框的宽度和高度 更大
                nimg_x = img[int(y - h1 / 2):int(y + h1 / 2), int(x - w1 / 2):int(x + w1 / 2)]
                nimg_x = cv2.resize(nimg_x, (sliceResize[0], sliceResize[1])).reshape(sliceResize + [1])
                # cv2.imshow(" ",nimg_x)
                vertebraSlice[imgPath].append(nimg_x)
    np.savez(r"testDiscSlice.npz", dic=discSlice)
    np.savez(r"testVertebraSlice.npz", dic=vertebraSlice)

classNamedic1 = {"v1":"Normal","v2":"Degeneration"}
classNamedic2 = {"v1":"Normal","v2":"Bulge","v3":"Protruded","v4":"Extruded","v5":"Schmor"}

def xwhconvert(images,target=None,imageName=None,f=None,fc=None,train=True):
    '''
    功能1：准备train.txt OCdevkit\VOC2007\JPEGImages/study0_image37.jpg 124.88,176.88,135.12,187.12,1
              val.txt
    功能2：准备分类切片 imgesSlice_vertebra,imgesSlice_disc
    '''
    def findIndex(src,target):
        for index,value in enumerate(src):
            if value == target:
                return index

    img_x = images[imageName]

    m, n = img_x.shape

    tm = 11 #12
    tn = 11

    w,h = m/tm,n/tn #框的宽度和高度


    # print(m,n)
    datas = target[imageName]
    #修改--->椎间盘:1，椎体为：0
    for data in datas:
        x, y = data["coord"]
        identification = data["tag"]["identification"]
        if len(identification) < 3:
            k = data["tag"]["vertebra"].split(",")[0]

            kind = 0
            if len(k)>1:
                # nimg_x = img_x[int(y - h1 / 2):int(y + h1 / 2), int(x - w1 / 2):int(x + w1 / 2)]
                # nimg_x = cv2.resize(nimg_x,(sliceResize[0],sliceResize[1])).reshape(sliceResize+[1])
                # imgesSlice_vertebra.append(nimg_x)
                # ySlice_vertebra.append(yc)
                fc.write(" " + k +"_"+"0"+ ","+ "vertebra" + "," + str(x) + "," + str(y))
            else:
                print("文件名：",imageName,"未标记类别")
        else:
            lable = data["tag"]["disc"].split(",") #椎间盘要求更长而不是更高
            if len(lable) > 1: #存在两个标签
                k = lable[0]
                k2 = lable[1]
            else:
                k = lable[0]
                k2 = "0"
            kind = 1
            if len(k) > 1:
                # nimg_x = img_x[int(y - h1 / 2):int(y + h1 / 2), int(x - w1 / 2):int(x + w1 / 2)]
                # nimg_x = cv2.resize(nimg_x, (sliceResize[0], sliceResize[1])).reshape(sliceResize + [1])
                # imgesSlice_disc.append(nimg_x)
                # ySlice_disc.append(yc)
                fc.write(" " + k +"_"+k2+ ","+ "disc" + "," + str(x) + "," + str(y))
            else:
                print("文件名：",imageName,"未标记类别")
        index = kind
        # f.write(" "+str(x*dw)+","+str(y*dh)+","+str(w*dw)+","+str(h*dh)+"," + str(index)) #归一化
        f.write(" " + str(x-w/2) + "," + str(y - h/2) + "," + str(x + w/2) + "," + str(y + h/2) + "," + str(index))

        # plt.figure()
        # plt.imshow(img_x)
        # currentAxis = plt.gca()
        # rect = patches.Rectangle((x-w/2,y - h/2),w,h,linewidth=1,edgecolor='r',facecolor='none') #左上角的点
        # currentAxis.add_patch(rect)
        # plt.scatter(x, y,s=6,c='r')
        # plt.show()
        # print( )

    # imgesSlice_vertebra, ySlice_vertebra, imgesSlice_disc, ySlice_disc = np.array(imgesSlice_vertebra),np.array(ySlice_vertebra),\
    #                                                                      np.array(imgesSlice_disc),np.array(ySlice_disc)
    # return imgesSlice_vertebra,ySlice_vertebra,imgesSlice_disc,ySlice_disc

def scanFeature(templete,img):
    pass

def dealImg(image,y1,y2,x1,x2,sliceResize):
    if y1<=0:
        y1 = 0
    if x1<=0:
        x1 = 0
    img = image[y1:y2,x1:x2]
    nimg_x = cv2.resize(img, (sliceResize[0], sliceResize[1]))
    return nimg_x[:, :, 1].reshape(sliceResize + [1])

def getTrinClf(dataTxt,flag=None,sliceResize=None): #train val
    f = open(dataTxt,'r')

    datas = f.readlines()

    disc_label = ['v' + str(i) for i in range(1, 5)]
    disc_labelv5 = ['0', "v5"]
    v15count = {"v1":0,"v5":0}
    v14count = {"v1":0,"v2":0,"v3":0,"v4":0}
    vertebracount = {"v1":0,"v2":0}
    vertebraSlice = {}
    discSlice = {}

    imgesSlice_vertebra_train,ySlice_vertebra_train = [],[]
    imgesSlice_disc_train, ySlice_disc_train = [], []
    imgesSlice_discv5_train, ySlice_discv5_train = [], []


    for data in datas:
        d = data.split()
        imgPath = d[0]
        img = cv2.imread(imgPath)

        m,n,c = img.shape

        mean = np.mean(img)
        var = np.mean(np.square(img - mean))
        imgNorm = (img - mean) / np.sqrt(var)

        xy = np.array(list(map(lambda x:list(map(int,x.split(',')[2:])), d[1:]))) #从小到大
        xy = sorted(xy,key=lambda x:x[1])
        deta = []
        for i in range(len(xy)-1):
            deta.append(xy[i+1][1]-xy[i][1])
        deta = np.mean(deta)
        print("deta:",deta)
        miny_pre = 0

        imgesSlice_vertebra = []
        ySlice_vertebra = []
        imgesSlice_disc = []
        ySlice_disc = []
        imgesSlice_discv5 = []
        ySlice_discv5 = []

        for target in d[1:]:
            target = target.split(',')
            x, y = list(map(int, target[2:]))

            if target[1] == 'disc':
                labels = target[0].split('_')

                if labels[0] in disc_label and labels[1] in disc_labelv5:
                    v14 = labels[0]
                    v5 = labels[1]
                elif labels[1] in disc_label and labels[0] in disc_labelv5:
                    v14 = labels[1]
                    v5 = labels[0]

                w1, h1 = m / 5, n / 20  # 椎间盘要求更长而不是更高  n控制高度
                offset = 5

                miny = y - h1
                maxy = y + h1
                minx = x + offset - w1/2
                maxx = x + offset + w1/2
                if miny < 0:
                    miny = 0
                if maxy > m:
                    maxy = m
                if minx < 0 :
                    minx = 0
                if maxx > n:
                    maxx = n
                if miny > m-10 or minx > n-10:
                    break

                xlen2 = int(w1 / 2)


                # nimg_x_Norm = imgNorm[int(miny):int(maxy), int(minx):int(maxx)]
                # nimg_x_r_Norm = nimg_x_Norm[:, int(xlen2):]
                # nimg_x_r_Norm = np.where(nimg_x_r_Norm > imgNorm[y,x,0], nimg_x_r_Norm, 0)
                # nimg_x_r_Norm = cv2.resize(nimg_x_r_Norm, (sliceResize[0], sliceResize[1]))

                X = dealImg(img,
                            int(miny), int(maxy), int(minx) + int(xlen2), int(maxx),
                            sliceResize)

                imgesSlice_disc.append(X)
                ySlice_disc.append(v14)
                v14count[v14] += 1

                r = np.random.randint(0,10,1)[0]

                if v14 == "v1":
                    if r > 7:
                        ry1 = np.random.randint(1,10,1)[0]
                        ry2 = np.random.randint(1,10,1)[0]

                        rx1 = np.random.randint(1,20,1)[0]
                        rx2 = np.random.randint(1,20,1)[0]
                        X = dealImg(img,
                                    int(miny)-ry1, int(maxy)+ry2, int(minx)+ int(xlen2)-rx1, int(maxx) +rx2,
                                    sliceResize)

                        imgesSlice_disc.append(X)
                        ySlice_disc.append(v14)
                        v14count[v14] +=1
                else:
                    ry1 = np.random.randint(1, 10, 1)[0]
                    ry2 = np.random.randint(1, 10, 1)[0]

                    rx1 = np.random.randint(1, 20, 1)[0]
                    rx2 = np.random.randint(1, 20, 1)[0]
                    X = dealImg(img,
                                int(miny) - ry1, int(maxy) + ry2, int(minx) + int(xlen2) - rx1, int(maxx) + rx2,
                                sliceResize)

                    imgesSlice_disc.append(X)
                    ySlice_disc.append(v14)
                    v14count[v14] += 1

                #     plt.subplot(4, 5, int(i+5*1))
                #     plt.title(v14)
                #     plt.imshow(img[int(miny)-ry1: int(maxy)+ry2, int(minx)+ int(xlen2)-rx1:int(maxx)+rx2])
                # plt.show()

                # nimg_x_r_zero = np.concatenate([zero_left,nimg_x_r],axis=1)

                # print("对右侧做数据增强")
                # if v14 in ["v2","v3","v4"]:
                #     for ii in range(10,40,20):
                #         contrast = 1  # 对比度
                #         brightness = ii  # 亮度
                #         nimg_x_brightness = cv2.addWeighted(nimg_x_r, contrast, nimg_x_r, 0, brightness)
                #         X = nimg_x_brightness[:, :, 1].reshape(sliceResize + [1])
                #         imgesSlice_disc.append(X)
                #         ySlice_disc.append(v14)
                #         v14count[v14] += 1


                # scanFeature(img[int(x-6):int(x+6), int(y-6):int(y+6)])
                #
                # plt.subplot(4, 5, int(1+5*1))
                # plt.title(v14)
                # plt.imshow(nimg_x)
                #
                # knn = cv2.createBackgroundSubtractorKNN(detectShadows = True)
                # nimg_x_r1 = knn.apply(nimg_x_r.copy())
                # plt.subplot(4, 5, int(2+5*1))
                # plt.title(v14)
                # plt.imshow(nimg_x_r1)
                #
                # plt.subplot(4, 5, int(3+5*1))
                # plt.title(v14)
                # plt.imshow(nimg_x_r)
                #
                # plt.subplot(4, 5, int(4+5*1))
                # plt.title(v14)
                # plt.imshow(nimg_x_r_Norm)
                # plt.show()

                #椎间盘要做一个三块的切片：保留 右半部，上半部，下半部

                nimg_x = img[int(miny):int(maxy), int(minx):int(maxx)]
                nimg_x = cv2.resize(nimg_x, (sliceResize[0], sliceResize[1]))
                if v5 == "v5": #如果等于v5 做数据增强

                    print("v5 数据增强~~")
                    # plt.figure(1)
                    j = 1
                    # for ii in range(10,20,20):
                    #     contrast = 1  # 对比度
                    #     brightness = ii  # 亮度
                    #     nimg_x_brightness = cv2.addWeighted(nimg_x, contrast, nimg_x, 0, brightness)
                    #     X = nimg_x_brightness[:, :, 1].reshape(sliceResize + [1])
                    #     imgesSlice_discv5.append(X)
                    #     ySlice_discv5.append(v5)  # 0:非椎体疝出 1：椎体疝出
                    #     v15count['v5'] += 1
                    #
                    #     # 变亮 水平垂直
                    #     nimg_x_flip = cv2.flip(nimg_x_brightness, -1)
                    #     X = nimg_x_flip[:, :, 1].reshape(sliceResize + [1])
                    #     imgesSlice_discv5.append(X)
                    #     ySlice_discv5.append(v5)  # 0:非椎体疝出 1：椎体疝出
                    #     v15count['v5'] += 1
                    #
                    #     nimg_x_flip = cv2.flip(nimg_x_brightness, 0)
                    #     X = nimg_x_flip[:, :, 1].reshape(sliceResize + [1])
                    #     imgesSlice_discv5.append(X)
                    #     ySlice_discv5.append(v5)  # 0:非椎体疝出 1：椎体疝出
                    #     v15count['v5'] += 1
                    #     plt.subplot(3,2,j)
                    #     plt.title(v5)
                    #     plt.imshow(nimg_x_flip)
                    #     j += 1
                    # plt.show()
                    height, width, c = nimg_x.shape

                    # for ii in range(-15,15,4):
                    #     angle = ii
                    #     M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
                    #     nimg_x_rotation = cv2.warpAffine(nimg_x, M, (height, width))
                    #     X = nimg_x_rotation[:, :, 1].reshape(sliceResize + [1])
                    #     imgesSlice_discv5.append(X)
                    #     ySlice_discv5.append(v5)  # 0:非椎体疝出 1：椎体疝出
                    #     v15count['v5'] += 1

                    nimg_x_flip = cv2.flip(nimg_x, 1)
                    X = nimg_x_flip[:, :, 1].reshape(sliceResize + [1])
                    imgesSlice_discv5.append(X)
                    ySlice_discv5.append(v5)  # 0:非椎体疝出 1：椎体疝出
                    v15count['v5'] += 1

                    X = nimg_x[:, :, 1].reshape(sliceResize + [1])
                    imgesSlice_discv5.append(X)
                    ySlice_discv5.append(v5)  # 0:非椎体疝出 1：椎体疝出
                    v15count['v5'] += 1
                else:
                    X = nimg_x[:, :, 1].reshape(sliceResize + [1])
                    imgesSlice_discv5.append(X)
                    ySlice_discv5.append(v5) # 0:非椎体疝出 1：椎体疝出
                    v15count['v1'] += 1

            else:
                labels = target[0].split('_')
                v12 = labels[0]
                v5 = labels[1]
                w1, h1 = m / 5, n / 10  # 识别框的宽度和高度 更大
                miny = y - h1
                maxy = y + h1
                minx = x - w1 / 2
                maxx = x + w1 / 2

                if miny < 0:
                    miny = 0
                if minx < 0:
                    minx = 0
                nimg_x = img[int(miny):int(maxy), int(minx):int(maxx)]

                # nimg_x_u = nimg_x[:ylen2, :]
                # nimg_x_u = cv2.resize(nimg_x_u, (sliceResize[0], sliceResize[1]))
                # nimg_x_d = nimg_x[ylen2:, :]
                # nimg_x_d = cv2.resize(nimg_x_d, (sliceResize[0], sliceResize[1]))

                # 椎柱要做一个两块的切片：保留 上半部，下半部,由于识别率比较高，所以先暂时不做处理
                nimg_x = cv2.resize(nimg_x, (sliceResize[0], sliceResize[1]))

                if v12 == "v1":
                    imgesSlice_vertebra.append(nimg_x[:, :, 1].reshape(sliceResize + [1]))
                    ySlice_vertebra.append(v12)  # 0:非椎体疝出 1：椎体疝出
                    vertebracount[v12] += 1

                    nimg_x_flip = cv2.flip(nimg_x, 1)
                    X = nimg_x_flip[:, :, 1].reshape(sliceResize + [1])
                    imgesSlice_vertebra.append(X)
                    ySlice_vertebra.append(v12)  # 0:非椎体疝出 1：椎体疝出
                    vertebracount[v12] += 1

                if v12 == "v2":
                    ri = np.random.randint(1,4)
                    X = nimg_x[:, :, 1].reshape(sliceResize + [1])
                    imgesSlice_vertebra.append(X)
                    ySlice_vertebra.append(v12)
                    vertebracount[v12] += 1

                    # 想做一个平衡数据
                    # if len(imgesSlice_vertebra) >= 1:
                    #     if ri%2 == 0:
                    #         X = nimg_x[:, :, 1].reshape(sliceResize + [1])
                    #         imgesSlice_vertebra.append(X)
                    #         ySlice_vertebra.append(v12)
                    #         vertebracount[v12] += 1
                    # else:
                    #     X = nimg_x[:, :, 1].reshape(sliceResize + [1])
                    #     imgesSlice_vertebra.append(X)
                    #     ySlice_vertebra.append(v12)
                    #     vertebracount[v12] += 1


        imgesSlice_vertebra_train.append(np.array(imgesSlice_vertebra))
        ySlice_vertebra_train.append(np.array(ySlice_vertebra))

        imgesSlice_disc_train.append(np.array(imgesSlice_disc))
        ySlice_disc_train.append(np.array(ySlice_disc))

        imgesSlice_discv5_train.append(imgesSlice_discv5)
        ySlice_discv5_train.append(ySlice_discv5)

    plt.subplot(1,3,1)
    plt.title("vertebra")
    plt.bar(list(vertebracount.keys()),list(vertebracount.values()))
    plt.subplot(1,3,2)
    plt.title("v14count")
    plt.bar(list(v14count.keys()),list(v14count.values()))
    plt.subplot(1,3,3)
    plt.title("v1v5")
    plt.bar(list(v15count.keys()),list(v15count.values()))
    plt.show()

    np.save(r"data2class2/imgesSlice_vertebra_{}.npy".format(flag),np.array(imgesSlice_vertebra_train)) # v12
    np.save(r"data2class2/ySlice_vertebra_{}.npy".format(flag), np.array(ySlice_vertebra_train))

    np.save(r"data2class2/imgesSlice_disc_{}.npy".format(flag), np.array(imgesSlice_disc_train)) # v14
    np.save(r"data2class2/ySlice_disc_{}.npy".format(flag), np.array(ySlice_disc_train))

    np.save(r"data2class2/imgesSlice_discv5_{}.npy".format(flag), np.array(imgesSlice_discv5_train))  # v5
    np.save(r"data2class2/ySlice_discv5_{}.npy".format(flag), np.array(ySlice_discv5_train))

    f.close()


def splice2classification():
    images = np.load(r"image.npz", allow_pickle=True)["dic"][()]
    target = np.load(r"target.npz", allow_pickle=True)["dic"][()]
    pass

def ready(dataPath,jsonPath):
    with open(jsonPath,"r",encoding="utf-8") as f:
        jsonTarge = json.loads(f.read())
    f.close()
    step1(dataPath,jsondata=jsonTarge)
    img2jpg()

def test():
    dataPath = r"H:\dataBase\tianchi_spinal\lumbar_train150"
    jsonPath = r"H:\dataBase\tianchi_spinal\lumbar_train150_annotation.json"
    dataTestPath = r"H:\dataBase\tianchi_spinal\lumbar_testA50"

    step1Test(dataPath=dataTestPath)
    # ready(dataPath, jsonPath)
    pass

if __name__ =="__main__":
    test()


    sys.exit()

    # #-------------------step1--------------------   预处理 51
    dataPath = r"H:\dataBase\tianchi_spinal\lumbar_train51\train"
    jsonPath = r"H:\dataBase\tianchi_spinal\lumbar_train51_annotation.json"
    #
    with open(jsonPath,"r",encoding="utf-8") as f:
        jsonTarge = json.loads(f.read())
    f.close()
    print(len(jsonTarge))

    step1(dataPath,jsondata=jsonTarge)

    # #-------------------step2--------------------   将数据以图像和html形式保存，数据集准备阶段
    img2jpg()  #保存数据为jpg

    sys.exit()

    #------------------------分析矩形--------------------
    #-------------------step1--------------------   预处理 150
    # dataPath = r"H:\dataBase\tianchi_spinal\lumbar_train150"
    # jsonPath = r"H:\dataBase\tianchi_spinal\lumbar_train150_annotation.json"
    # #
    # with open(jsonPath,"r",encoding="utf-8") as f:
    #     jsonTarge = json.loads(f.read())
    # f.close()
    # print(len(jsonTarge))
    #
    # step1(dataPath,jsondata=jsonTarge)
    # #
    # sys.exit()
    # studyUid='0020|000d',instanceUid='0008|0018',seriesUid='0020|000e'


    images = np.load(r"image.npz",allow_pickle=True)["dic"][()]
    target = np.load(r"target.npz",allow_pickle=True)["dic"][()]

    #vertebra :椎体 red disc：椎间盘 green
    font = cv2.FONT_HERSHEY_SIMPLEX

    print(len(images))

    for imageId in images.keys():
        img_x = images[imageId]
        img_x = cv2.cvtColor(img_x,cv2.COLOR_GRAY2BGR)
        m, n ,c = img_x.shape

        print(m,n,c)
        datas = target[imageId]
        print()

        img_x_2 = totarget(datas, img_x, 1, 1)
        cv2.imshow('0', img_x)
        cv2.waitKey(0)

    pass