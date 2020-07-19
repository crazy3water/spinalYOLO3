'''
第二步：优化中心点：
    利用回归模型优化中心点
为第三步分类做准备
    再次使用矩形框对图像切片
'''
import numpy as np
import cv2
from sklearn.linear_model import Ridge,LogisticRegression,Lasso
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


    #  图像是以左上角为原点，所以y取最大值是最下面 ---- 可视化分析
    # for coord,tag,img in zip(coords,tags,images):
    #     img = cv2.imread(img)
    #     coord = np.array(coord)
    #     minIndex = np.argmax(coord[:,1])
    #     img = cv2.circle(img,(coord[minIndex][0],coord[minIndex][1]),radius=3,color=[0,0,255],thickness=1)
    #     for x,y in coord:
    #         print("x:",x,"y:",y)
    #         img = cv2.circle(img, (x, y), radius=1, color=[0, 0, 255], thickness=-1)
    #     print("最低位x:{},y:{},类型：{}".format(coord[minIndex][0],coord[minIndex][1],tag[minIndex]))
    #     cv2.imshow(' ',img)
    #     cv2.waitKey(0)


def lasso_regression(train,test):
    # 图像不一样大小的时候，导致像素点是对应不上的，所以现在把图像和像素点 放缩
    #拟合思路：
    # clf = LogisticRegression(C=0.1,penalty='l2')
    # clf = Lasso(alpha=0.1)
    clf = Ridge(alpha=0.1)
    polyFeacture = PolynomialFeatures(degree=2)
    trainX = train[:,1].reshape([-1,1])  #[index,y,x]
    trainy = train[:,-1]
    clf.fit(polyFeacture.fit_transform(trainX),trainy)
    testX = test[:,1].reshape([-1,1])
    testy = test[:,-1]
    pre = clf.predict(polyFeacture.fit_transform(testX))
    #---------------可视化回归分析---------------
    # plt.figure(0)
    # for i in range(test.shape[0]):
    #     plt.scatter(train[i * 11:i * 11+11,2],train[i*11:i*11+11,1])
    #     plt.scatter(test[i, 2], test[i, 1],c='r')
    #     plt.scatter(test[i, 2], pre[i], c='g')
    #     for j in np.arange(i * 11,i * 11+11):
    #         plt.text(train[j,2],train[j,1],str(train[j,0]))
    #     plt.text(test[i, 2], test[i, 1],test[i, 0])
    #     # plt.waitforbuttonpress()
    #     plt.show()

    print("loss:",np.linalg.norm(pre-testy,ord=2))
    return clf



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





def computDis(coord,targetX):
    sum = 0
    for i in range(len(coord)):
        x, y = coord[i][0], coord[i][1]
        sum += abs(x-targetX)
    return sum


def delAway(coords,tags,images):
    #删除 偏离的 点
    #重复 在6个像素点内
    ncoords, ntags = [], []
    for coord,tag,img in zip(coords,tags,images):
        if len(coord) > 5:
            #将坐标点排序
            coord = np.array(coord)
            minIndex = np.argsort(coord[:,1])

            coord = coord[minIndex]
            tag = np.array(tag)[minIndex]

            recordy = []
            for i in range(len(coord)-1):
                recordy.append(abs(coord[i+1][1] - coord[i][1]))
            recordy = np.array(recordy).reshape([-1])
            zwy = np.median(recordy)

            delIndex = 1
            while(delIndex):
                for i in range(len(coord)-1):
                    if abs(coord[i+1][1] - coord[i][1]) < zwy*0.5 and \
                            abs(coord[i + 1][0] - coord[i][0]) < zwy*0.5:
                        coord[i][1],coord[i][0] = int((coord[i+1][1] + coord[i][1]) /2),int((coord[i+1][0] + coord[i][0]) /2)
                        coord = np.delete(coord,i+1,axis=0)
                        tag = np.delete(tag,i+1,axis=0)
                        delIndex = 1
                        print("删除{}重合点,y差值中位数为：".format(img),zwy,"*0.5=",zwy*0.5)
                        break
                delIndex =0

            study, imgN = img.split('/')[-1].split('.')[0].split('_')
            if study == "study214":
                print()

            # d1 = throldCluster(coord[:,0])
            # coord = np.delete(coord, d1, axis=0)
            # tag = np.delete(tag, d1, axis=0)
            # ------------------x------------------
            record = []
            data = coord[:,0]
            for i in range(len(data)):
                record.append(computDis(coord,data[i]))

            delflag = 1
            #每个点x到其他点x的距离之和
            dis = record
            d1 = []
            i = len(dis)-1
            while(i>=0):
                targetDis = dis[i]
                if targetDis > 2*np.mean(dis[:i]+dis[i:]):
                    print("{}第{}个点x偏离".format(img,i))
                    d1.append(i)
                    print(data)
                    print(np.percentile(data, 0.8))
                    i -= 2
                    # break
                else:
                    delflag = 0
                    i -= 1

            coord = np.delete(coord, d1, axis=0)
            tag = np.delete(tag, d1, axis=0)
            #------------------y------------------
            record = []
            data = coord[:,1]
            for i in range(len(data)):
                record.append(computDis(coord,data[i]))

            dis = record
            d1 = []
            i = len(dis)-1
            while(i>=0):
                targetDis = dis[i]
                if targetDis > 2.0*np.mean(dis[:i]+dis[i:]):
                    print("{}第{}个点y偏离".format(img,i))
                    d1.append(i)
                    print(data)
                    print(np.percentile(data, 0.8))
                    i -= 2
                    # break
                else:
                    i -= 1

            coord = np.delete(coord, d1, axis=0)
            tag = np.delete(tag, d1, axis=0)
            # ------------------y------------------
            ncoords.append(coord)
            ntags.append(tag)
    return ncoords,ntags,images

def midPoint(coords,tags,images):
    ncoords, ntags = [], []
    for coord, tag, img in zip(coords,tags,images):
        #将坐标点排序
        coord = np.array(coord)
        minIndex = np.argsort(coord[:,1])

        coord = coord[minIndex]
        tag = np.array(tag)[minIndex]

        recordy = []
        for i in range(len(coord)-1):
            recordy.append(coord[i+1][1] - coord[i][1])
        recordy = np.array(recordy).reshape([-1])
        zwy = np.median(recordy)

        delIndex = 1
        while(delIndex):
            for i in range(len(coord)-1):
                if abs(coord[i+1][1] - coord[i][1]) > zwy*1.8: #y之间有缝隙
                    newPoint = np.array([int((coord[i+1][0] + coord[i][0]) /2),int((coord[i+1][1] + coord[i][1]) /2)]).reshape([1,2])
                    if tag[i] == "vertebra":
                        newtag = np.array(["disc"])
                    else:
                        newtag = np.array(["vertebra"])

                    coord = np.concatenate([coord,newPoint],axis=0)
                    tag = np.concatenate([tag,newtag],axis=0)
                    delIndex = 0
                    print("修补{}缺失点,y差值中位数为：".format(img),zwy,"*1.8=",zwy*1.8)
                    break
                delIndex =0
        ncoords.append(coord)
        ntags.append(tag)
    return ncoords,ntags,images



if __name__ == "__main__":
    train_regression2test(getResultPath=r"resultStep1_op.txt",train=False)
    pass