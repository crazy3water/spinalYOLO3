import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from readyData import dicom_metainfo
import SimpleITK as sitk
import re

def step1Test(dataPath):
    '''
    在准备jpg时，首先先判断是否为T2_sag，只对T2_sag做定位识别
    target: t2sag:0, t1sag:1, t2tra:2
    :param dataPath:
    :return:
    '''
    count = 0
    study = os.listdir(dataPath)

    def parse_orentation(data):
        r2,c2 = [],[]
        for i in data.split("\\")[:3]:
            r2.append(float(i))
        for i in data.split("\\")[3:]:
            c2.append(float(i))
        return np.array(r2),np.array(c2)

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

    for studyI in study:
        Impath = os.listdir(os.path.join(dataPath, studyI))
        print("study:", studyI)
        for Impathi in Impath:
            image_file_reader = sitk.ImageFileReader()
            image_file_reader.SetImageIO('GDCMImageIO')
            image_file_reader.SetFileName(os.path.join(os.path.join(dataPath, studyI), Impathi))
            try:
                image_file_reader.ReadImageInformation()
            except:
                break

            description = image_file_reader.GetMetaData('0008|103e')
            if re.match(r"(.*)[tT]2(.*)[sS][aA][gG]", description, flags=0) or re.match(
                    r"(.*)[sS][aA][gG](.*)[tT]2(.*)", description, 0):

                orienta, pos, pixelSpacingXY = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi),
                                                              ['0020|0037', '0020|0032', '0028|0030'])
                (r1, c1), o1, pixelSpacingXY = parse_orentation(orienta), parse_position(pos), parse_pixelSpacing(
                    pixelSpacingXY)

                image1 = image_file_reader.Execute()
                if image1.GetNumberOfComponentsPerPixel() == 1:
                    image1 = sitk.RescaleIntensity(image1, 0, 255)
                if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
                    image1 = sitk.InvertIntensity(image1, maximum=255)
                image1 = sitk.Cast(image1, sitk.sitkUInt8)
                image1 = sitk.GetArrayFromImage(image1)[0]
                srcHeigh, srcWidth = image1.shape[:2]
                print(srcHeigh, srcWidth)
                n1 = np.cross(r1, c1)

                p1 = o1  # 左上角的点
                p2 = p1 + r1 * pixelSpacingXY [0] * (srcWidth - 1)  # 右上角的点
                p3 = p2 + c1 * pixelSpacingXY [1] * (srcHeigh - 1)  # 右下角的点
                p4 = p1 + c1 * pixelSpacingXY [1] * (srcHeigh - 1)  # 左下角的点


                # for p in [p1, p2, p3, p4]:
                #     # 到方向向量的投影
                #     xmap, ymap = np.dot(p - o1, r1) / np.linalg.norm(r1), np.dot(p - o1, c1) / np.linalg.norm(c1)
                #     # 转为pixel坐标
                #     xmap /= pixelSpacingXY [0]
                #     ymap /= pixelSpacingXY [1]
                #     plt.plot(int(xmap), int(ymap), "o", color='red', ms=10)
                #     plt.imshow(image, cmap="gray")
                #     plt.show()
                #     print('d')
            if re.match(r"(.*)[tT][rR][aA](.*)", description, flags=0) or re.match(
                    r"(.*)[tT][rR][aA](.*)", description, 0):
                plt.imshow(image1, cmap="gray")
                orienta, pos, pixelSpa   = dicom_metainfo(os.path.join(os.path.join(dataPath, studyI), Impathi),
                                                              ['0020|0037', '0020|0032', '0028|0030'])
                (r2, c2), o2, pixelSpa   = parse_orentation(orienta), parse_position(pos), parse_pixelSpacing(
                    pixelSpa  )

                image = image_file_reader.Execute()
                if image.GetNumberOfComponentsPerPixel() == 1:
                    image = sitk.RescaleIntensity(image, 0, 255)
                if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
                    image = sitk.InvertIntensity(image, maximum=255)
                image = sitk.Cast(image, sitk.sitkUInt8)
                image = sitk.GetArrayFromImage(image)[0]
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
                    continue

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
                for cp in cps:
                    # 看下算出来的点在不在定位图平面上 带入平面方程 正确应该是几乎等于0
                    print(n1 * (cp - o1))

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
                    coords_img.append(np.array([int(xmap / pixelSpacingXY[0]), int(ymap / pixelSpacingXY[1])]))


                plt.plot((coords_img[0][0], coords_img[1][0]), (coords_img[0][1], coords_img[1][1]))
                plt.plot(int(coords_img[0][0]), int(coords_img[0][1]), "o", color='red', ms=10)
                plt.plot(int(coords_img[1][0]), int(coords_img[1][1]), "o", color='red', ms=10)
                plt.show()




if __name__ == "__main__":
    step1Test(dataPath = r"H:\dataBase\tianchi_spinal\lumbar_train150")
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
