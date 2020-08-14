import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from readyData import dicom_metainfo,dicom2array
import scipy.cluster.hierarchy as hcluster
import re


def parse_orentation(data):
    r2, c2 = [], []
    for i in data.split("\\")[:3]:
        r2.append(float(i))
    for i in data.split("\\")[3:]:
        c2.append(float(i))
    return np.array(r2), np.array(c2)


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

def dealCluster(clusters,clusData):
    clusterDic = {}
    if len(clusters) > 2:
        for cluster,data in zip(clusters,clusData):
            if cluster not  in clusterDic.keys():
                clusterDic[cluster] = [data]
            else:
                clusterDic[cluster].append(data)

        for key,value in clusterDic.items():
            clusterDic[key] = value[int(len(value) / 2)]
    return clusterDic

def xyIndcm(x,y,clusterDic):
    for key,value in clusterDic.items():
        y1, y2, x1, x2, dcm = value
        k = (y1-y2)/(x1-x2)
        k1 = (y-y2)/(x-x2)
        if abs(k-k1) < 0.1:
            return True,dcm

    return False,None

def step1Test(dataPath,testTxt=None):
    '''
    在准备jpg时，首先先判断是否为T2_sag，只对T2_sag做定位识别
    target: t2sag:0, t1sag:1, t2tra:2
    :param dataPath:
    :return:
    '''
    count = 0
    study = os.listdir(dataPath)



    f = open(testTxt, 'r')
    datas = f.readlines()
    f.close()

    for data in datas:
        d = data.split()
        ImgPath = d[0].split('/')[-1].split('.')[0].split('_')  # 寻找dcm
        studyI = ImgPath[0]
        dicmPath = dataPath + '/' + ImgPath[0] + "/" + ImgPath[1] + '.dcm'  #answer中间帧
        Impath = os.listdir(os.path.join(dataPath, studyI)) #找到study下的所有dcm 为了寻找tra，同一标签下第一左边方差小
        describes2pos = {}
        describes2imgPathes = {}

        for Impathi in Impath:
            pos = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi), ['0020|0032'])[0]
            describ = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi), ['0008|103e'])[0]
            seriesNumber = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi), ['0020|0011'])[0]
            describ += seriesNumber
            if re.match(r"(.*)[sS][cC][oO][uU][tT](.*)", describ, 0): #不加入
                continue
            elif re.match(r"(.*)[sS][aA][gG](.*)",describ,flags=0) or re.match(r"(.*)[sS][aA][gG](.*)",describ,0):
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
            if re.match(r"(.*)[tT][rR][aA](.*)",key,flags=0):
                flagN = True
                describes2posN[key] = value
                describes2imgPathesN[key] = describes2imgPathes[key]
        if flagN:
            describes2pos = describes2posN
            describes2imgPathes = describes2imgPathesN
        plt.figure(1)
        imgSAG = dicom2array(dicmPath)
        plt.imshow(imgSAG)

        sagPoints = SagKeyPoints(dicmPath)
        index = 0
        for key,value in describes2pos.items():
            if np.std(np.array(value))<10 and len(value) > 6:
                # plt.figure(2)
                print("找到tra! -- ",studyI,key," -- ",len(value))
                count += 1
                clusData = []
                for traPath in describes2imgPathes[key]:
                    traDim = os.path.join(os.path.join(dataPath, studyI), traPath)
                    TraMapSagPoints = TraMapSag(sagPoints,traDim)
                    imgTRA = dicom2array(traDim)
                    # plt.subplot(2,int(len(value)/2)+1,index+1)
                    # plt.imshow(imgTRA)
                    plt.plot((TraMapSagPoints[0][0], TraMapSagPoints[1][0]), (TraMapSagPoints[0][1], TraMapSagPoints[1][1]))

                    x1,x2 = int(TraMapSagPoints[0][0]),int(TraMapSagPoints[1][0])
                    y1,y2 = int(TraMapSagPoints[0][1]),int(TraMapSagPoints[1][1])

                    clusData.append([y1,y2,x1,x2, traDim])
                    plt.plot(x1, y1, "o", color='red', ms=10)
                    plt.plot(x2, y2, "o", color='g', ms=10)
                    plt.plot(int((x1+x2)/2), int((y1+y2)/2), "o", color='g', ms=10)
                    plt.text(x1, y1,str(index))
                    index += 1


                thresh = 30
                print(sorted(clusData,key=lambda x:x[0]))
                clusData = sorted(clusData, key=lambda x: x[0])
                clusDataN = [[i[0]] for i in clusData]
                traDimN = [i[-1] for i in clusData]
                clusters = hcluster.fclusterdata(np.array(clusDataN), thresh, criterion="distance")
                print(clusters,set(clusters.tolist()))

                clustersDic = dealCluster(clusters,clusData)
                print(clustersDic)
                d = sortPoint(d)
                for index, target in enumerate(d):
                    x, y, vd, c, p = target.split(',')
                    if vd == "disc":
                        print("==")
                        flag,dicm = xyIndcm(int(x),int(y),clustersDic)
                        if flag:
                            print(dicm)
                            plt.plot(int(x), int(y), "o", color='r', ms=5)
                plt.show()
    print(count)





    # for studyI in study:
    #     Impath = os.listdir(os.path.join(dataPath, studyI))
    #     print("study:", studyI)
    #     for Impathi in Impath:
    #
    #         pathDim = os.path.join(os.path.join(dataPath, studyI), Impathi)
    #         image_file_reader = sitk.ImageFileReader()
    #         image_file_reader.SetImageIO('GDCMImageIO')
    #         image_file_reader.SetFileName(pathDim)
    #         try:
    #             image_file_reader.ReadImageInformation()
    #         except:
    #             break
    #
    #         description = image_file_reader.GetMetaData('0008|103e')
    #         if re.match(r"(.*)[tT]2(.*)[sS][aA][gG]", description, flags=0) or re.match(
    #                 r"(.*)[sS][aA][gG](.*)[tT]2(.*)", description, 0):
    #
    #             orienta, pos, pixelSpacingXY = dicom_metainfo(pathDim,
    #                                                           ['0020|0037', '0020|0032', '0028|0030'])
    #             (r1, c1), o1, pixelSpacingXY = parse_orentation(orienta), parse_position(pos), parse_pixelSpacing(
    #                 pixelSpacingXY)
    #
    #             image1 = image_file_reader.Execute()
    #             if image1.GetNumberOfComponentsPerPixel() == 1:
    #                 image1 = sitk.RescaleIntensity(image1, 0, 255)
    #             if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
    #                 image1 = sitk.InvertIntensity(image1, maximum=255)
    #             image1 = sitk.Cast(image1, sitk.sitkUInt8)
    #             image1 = sitk.GetArrayFromImage(image1)[0]
    #             srcHeigh, srcWidth = image1.shape[:2]
    #             print(srcHeigh, srcWidth)
    #             n1 = np.cross(r1, c1)
    #
    #             p1 = o1  # 左上角的点
    #             p2 = p1 + r1 * pixelSpacingXY [0] * (srcWidth - 1)  # 右上角的点
    #             p3 = p2 + c1 * pixelSpacingXY [1] * (srcHeigh - 1)  # 右下角的点
    #             p4 = p1 + c1 * pixelSpacingXY [1] * (srcHeigh - 1)  # 左下角的点
    #
    #
    #             # for p in [p1, p2, p3, p4]:
    #             #     # 到方向向量的投影
    #             #     xmap, ymap = np.dot(p - o1, r1) / np.linalg.norm(r1), np.dot(p - o1, c1) / np.linalg.norm(c1)
    #             #     # 转为pixel坐标
    #             #     xmap /= pixelSpacingXY [0]
    #             #     ymap /= pixelSpacingXY [1]
    #             #     plt.plot(int(xmap), int(ymap), "o", color='red', ms=10)
    #             #     plt.imshow(image, cmap="gray")
    #             #     plt.show()
    #             #     print('d')
    #         if re.match(r"(.*)[tT][rR][aA](.*)", description, flags=0) or re.match(
    #                 r"(.*)[tT][rR][aA](.*)", description, 0):
    #             plt.imshow(image1, cmap="gray")
    #             orienta, pos, pixelSpa   = dicom_metainfo(pathDim,
    #                                                           ['0020|0037', '0020|0032', '0028|0030'])
    #             (r2, c2), o2, pixelSpa   = parse_orentation(orienta), parse_position(pos), parse_pixelSpacing(
    #                 pixelSpa  )
    #
    #             image = image_file_reader.Execute()
    #             if image.GetNumberOfComponentsPerPixel() == 1:
    #                 image = sitk.RescaleIntensity(image, 0, 255)
    #             if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
    #                 image = sitk.InvertIntensity(image, maximum=255)
    #             image = sitk.Cast(image, sitk.sitkUInt8)
    #             image = sitk.GetArrayFromImage(image)[0]
    #             srcHeigh, srcWidth = image.shape[:2]
    #             p1 = o2  # 左上角的点
    #             p2 = p1 + r2 * pixelSpa[0] * (srcWidth - 1)  # 右上角的点
    #             p3 = p2 + c2 * pixelSpa[1] * (srcHeigh - 1)  # 右下角的点
    #             p4 = p1 + c2 * pixelSpa[1] * (srcHeigh - 1)  # 左下角的点
    #             # 点到平面的距离
    #             # p1 O1
    #             dv1 = np.dot(p1 - o1, n1)
    #             # p2 O1
    #             dv2 = np.dot(p2 - o1, n1)
    #             # p3 O1
    #             dv3 = np.dot(p3 - o1, n1)
    #             # p4 O1
    #             dv4 = np.dot(p4 - o1, n1)
    #
    #             iscross12 = ((dv1 > 0) & (dv2 < 0)) | ((dv1 < 0) & (dv2 > 0))
    #             iscross23 = ((dv2 > 0) & (dv3 < 0)) | ((dv2 < 0) & (dv3 > 0))
    #             iscross34 = ((dv3 > 0) & (dv4 < 0)) | ((dv3 < 0) & (dv4 > 0))
    #             iscross41 = ((dv4 > 0) & (dv1 < 0)) | ((dv4 < 0) & (dv1 > 0))
    #             if not (iscross12 | iscross23 | iscross34 | iscross41):
    #                 continue
    #
    #             cps = []
    #             for iscross, (pv, pp), (ddv, ddp) in zip([iscross12, iscross23, iscross34, iscross41],
    #                                                      [(p1, p2), (p2, p3), (p3, p4), (p4, p1)],
    #                                                      [(dv1, dv2), (dv2, dv3), (dv3, dv4), (dv4, dv1)]):
    #                 if not iscross:
    #                     # 是否相交
    #                     continue
    #                     # 计算交点坐标  相似三角形？
    #                 cp = [pv[i] + (pp[i] - pv[i]) * ddv / (ddv - ddp) for i in range(3)]
    #                 cps.append(np.array(cp))
    #             for cp in cps:
    #                 # 看下算出来的点在不在定位图平面上 带入平面方程 正确应该是几乎等于0
    #                 print(n1 * (cp - o1))
    #
    #             assert len(cps) == 2
    #             coords = []
    #             coords_img = []
    #             for cp in cps:
    #                 # 投影
    #                 xmap = np.dot(cp - o1, r1) / np.linalg.norm(r1)
    #                 ymap = np.dot(cp - o1, c1) / np.linalg.norm(c1)
    #
    #                 # 这个没什么用
    #                 coords.append(np.array(xmap, ymap))
    #                 # pixel坐标
    #                 coords_img.append(np.array([int(xmap / pixelSpacingXY[0]), int(ymap / pixelSpacingXY[1])]))
    #
    #
    #             plt.plot((coords_img[0][0], coords_img[1][0]), (coords_img[0][1], coords_img[1][1]))
    #             plt.plot(int(coords_img[0][0]), int(coords_img[0][1]), "o", color='red', ms=10)
    #             plt.plot(int(coords_img[1][0]), int(coords_img[1][1]), "o", color='red', ms=10)
    #             plt.show()

def SagKeyPoints(sagPath):
    image1 = dicom2array(sagPath)
    orienta, pos, pixelSpacingXY = dicom_metainfo(sagPath,
                                                  ['0020|0037', '0020|0032', '0028|0030'])
    (r1, c1), o1, pixelSpacingXY = parse_orentation(orienta), parse_position(pos), parse_pixelSpacing(
        pixelSpacingXY)

    srcHeigh, srcWidth = image1.shape[:2]
    n1 = np.cross(r1, c1)

    p1 = o1  # 左上角的点
    p2 = p1 + r1 * pixelSpacingXY[0] * (srcWidth - 1)  # 右上角的点
    p3 = p2 + c1 * pixelSpacingXY[1] * (srcHeigh - 1)  # 右下角的点
    p4 = p1 + c1 * pixelSpacingXY[1] * (srcHeigh - 1)  # 左下角的点

    return r1,c1,o1,pixelSpacingXY,n1

def TraMapSag(sagPoints,traPath):
    r1, c1, o1, pixelSpacingXY, n1 = sagPoints
    orienta, pos, pixelSpa = dicom_metainfo(traPath,
                                            ['0020|0037', '0020|0032', '0028|0030'])
    (r2, c2), o2, pixelSpa = parse_orentation(orienta), parse_position(pos), parse_pixelSpacing(
        pixelSpa)
    image = dicom2array(traPath)
    srcHeigh, srcWidth = image.shape[:2]
    p1 = o2  # 左上角的点
    p2 = p1 + r2 * pixelSpa[0] * (srcWidth - 1)  # 右上角的点
    p3 = p2 + c2 * pixelSpa[1] * (srcHeigh - 1)  # 右下角的点
    p4 = p1 + c2 * pixelSpa[1] * (srcHeigh - 1)  # 左下角的点
    # 点到平面的距离
    # p1 O1
    dv1 = np.dot(p1 - o1, n1)
    # p2 O1
    dv2 = np.dot(p2 - o1, n1)
    # p3 O1
    dv3 = np.dot(p3 - o1, n1)
    # p4 O1
    dv4 = np.dot(p4 - o1, n1)

    iscross12 = ((dv1 > 0) & (dv2 < 0)) | ((dv1 < 0) & (dv2 > 0))
    iscross23 = ((dv2 > 0) & (dv3 < 0)) | ((dv2 < 0) & (dv3 > 0))
    iscross34 = ((dv3 > 0) & (dv4 < 0)) | ((dv3 < 0) & (dv4 > 0))
    iscross41 = ((dv4 > 0) & (dv1 < 0)) | ((dv4 < 0) & (dv1 > 0))
    if not (iscross12 | iscross23 | iscross34 | iscross41):
        return

    cps = []
    for iscross, (pv, pp), (ddv, ddp) in zip([iscross12, iscross23, iscross34, iscross41],
                                             [(p1, p2), (p2, p3), (p3, p4), (p4, p1)],
                                             [(dv1, dv2), (dv2, dv3), (dv3, dv4), (dv4, dv1)]):
        if not iscross:
            # 是否相交
            continue
            # 计算交点坐标  相似三角形？
        cp = [pv[i] + (pp[i] - pv[i]) * ddv / (ddv - ddp) for i in range(3)]
        cps.append(np.array(cp))
    # for cp in cps:
    #     # 看下算出来的点在不在定位图平面上 带入平面方程 正确应该是几乎等于0
    #     print(n1 * (cp - o1))

    assert len(cps) == 2
    coords = []
    coords_img = []
    for cp in cps:
        # 投影
        xmap = np.dot(cp - o1, r1) / np.linalg.norm(r1)
        ymap = np.dot(cp - o1, c1) / np.linalg.norm(c1)

        # 这个没什么用
        coords.append(np.array(xmap, ymap))
        # pixel坐标
        coords_img.append(np.array([int(xmap / pixelSpacingXY[0]), int(ymap / pixelSpacingXY[1])])) #得到映射到sag的两个点
    return coords_img




if __name__ == "__main__":
    step1Test(dataPath = r"H:\dataBase\tianchi_spinal\lumbar_testA50",testTxt=r"resultStep3_.txt")
#     for slice_name in slice_names:
#         ## 获取每一张轴状图的信息
#         orienta, pos, pixelSpa = df_study.loc[
#             slice_name, ['orientation', "position", 'pixelSpacing', ]].values.tolist()
#         (r2, c2), o2, pixelSpa = parse_orentation(orienta), parse_position(pos), parse_pixelSpacing(pixelSpa)
#         image = np.array(
#             Image.open("data/images" + "/" + df_study.study.values[0] + "/" + slice_name.replace("dcm", "png")))
#         srcHeigh, srcWidth = image.shape[:2]
#
#         p1 = o2  # 左上角的点
#         p2 = p1 + r2 * pixelSpa[0] * (srcWidth - 1)  # 右上角的点
#         p3 = p2 + c2 * pixelSpa[1] * (srcHeigh - 1)  # 右下角的点
#         p4 = p1 + c2 * pixelSpa[1] * (srcHeigh - 1)  # 左下角的点
#         # 点到平面的距离
#         # p1 O1
#         dv1 = np.dot(p1 - o1, n1)
#         # p2 O1
#         dv2 = np.dot(p2 - o1, n1)
#         # p3 O1
#         dv3 = np.dot(p3 - o1, n1)
#         # p4 O1
#         dv4 = np.dot(p4 - o1, n1)
#         # 判断是否有交点
#         iscross12 = ((dv1 > 0) & (dv2 < 0)) | ((dv1 < 0) & (dv2 > 0))
#         iscross23 = ((dv2 > 0) & (dv3 < 0)) | ((dv2 < 0) & (dv3 > 0))
#         iscross34 = ((dv3 > 0) & (dv4 < 0)) | ((dv3 < 0) & (dv4 > 0))
#         iscross41 = ((dv4 > 0) & (dv1 < 0)) | ((dv4 < 0) & (dv1 > 0))
#         if not (iscross12 | iscross23 | iscross34 | iscross41):
#             continue
#     cps = []
#     for iscross, (pv, pp), (ddv, ddp) in zip([iscross12, iscross23, iscross34, iscross41],
#                                              [(p1, p2), (p2, p3), (p3, p4), (p4, p1)],
#                                              [(dv1, dv2), (dv2, dv3), (dv3, dv4), (dv4, dv1)]):
#         if not iscross:
#             # 是否相交
#             continue
#             # 计算交点坐标  相似三角形？
#         cp = [pv[i] + (pp[i] - pv[i]) * ddv / (ddv - ddp) for i in range(3)]
#         cps.append(np.array(cp))
#     for cp in cps:
#         # 看下算出来的点在不在定位图平面上 带入平面方程 正确应该是几乎等于0
#         print(n1 * (cp - o1))
#
#     assert len(cps) == 2
#     coords = []
#     coords_img = []
#     for cp in cps:
#         # 投影
#         xmap = np.dot(cp - o1, r1) / np.linalg.norm(r1)
#         ymap = np.dot(cp - o1, c1) / np.linalg.norm(c1)
#
#         # 这个没什么用
#         coords.append(np.array(xmap, ymap))
#         # pixel坐标
#         coords_img.append(np.array([int(xmap / pixelSpacingXY[0]), int(ymap / pixelSpacingXY[1])]))
#
#     if show:
#         plt.plot((coords_img[0][0], coords_img[1][0]), (coords_img[0][1], coords_img[1][1]))
# if show:
#     plt.show()
