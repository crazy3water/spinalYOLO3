'''
第三步：用于分类
训练:将训练集图像按中心点切割 标注
测试：将测试图像按第二步的结果进行切割，测试
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from yolo3 import densenet,Resnet
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
import tensorflow as tf

class CNN(object):
    def __init__(self,sliceResize,fileName=None):
        self.saveModel_vertebra = r"logs/001/"
        self.saveModel_disc = r"logs/002/"
        self.saveModel_discv5 = r"logs/004/"
        self.filename = fileName

        self.vertebra_model = None
        self.disc_model = None

        self.vertebra_label = ['v'+str(i) for i in range(1,3)]
        self.disc_label = ['v' + str(i) for i in range(1, 5)]
        self.disc_labelv5 = ['0',"v5"]
        self.resizeShape = sliceResize
        pass

    def npy2arry(self,data):
        if len(data) > 1:
            t = data[0]
            for i in range(1,len(data)):
                temp = data[i]
                t = np.concatenate([t,temp])
        else:
            t = data[0]
        return t

    def npy2y(self,y,classes):
        t = []
        for j in y:
            for i in j:
                t += [classes.index(i)]
        return t

    def CNNmodel(self,input,classes):
        x = keras.layers.Conv2D(filters=1,kernel_size= (2,2),strides=(1, 1))(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.MaxPooling2D((2,2),strides=(2,2))(x)

        x = keras.layers.Conv2D(filters=1,kernel_size= (2,2),strides=(1, 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.MaxPooling2D((2,2),strides=(2,2))(x)

        x = keras.layers.Conv2D(filters=1,kernel_size= (2,2),strides=(1, 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.MaxPooling2D((2,2),strides=(2,2))(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(classes, activation='softmax')(x)

        model = keras.Model(input,x)
        return model

    def denseNet(self,input,classes,inputshape):
        return densenet.DenseNet(img_input=input,blocks=[2,3,2,0],classes=classes,input_shape=inputshape)

    def ResNet(self,input,classes,inputshape):
        return Resnet.build_model(input,
                    input_shape=inputshape,
                    classes=classes)

    def train_vertebra_data(self):
        X = np.load(r"{}\imgesSlice_vertebra_train.npy".format(self.filename),allow_pickle=True)
        X = self.npy2arry(X)

        y = np.load(r"{}\ySlice_vertebra_train.npy".format(self.filename),allow_pickle=True)

        Xv = np.load(r"{}\imgesSlice_vertebra_val.npy".format(self.filename),allow_pickle=True)
        Xv = self.npy2arry(Xv)

        yv = np.load(r"{}\ySlice_vertebra_val.npy".format(self.filename),allow_pickle=True)

        classes = self.vertebra_label
        y = self.npy2y(y,classes)
        yv = self.npy2y(yv, classes)
        return X,y,Xv,yv

    def train_vertebra(self):
        X,y,Xv,yv = self.train_vertebra_data()

        inputLayer = keras.layers.Input(shape=X.shape[-3:])
        # model = self.CNNmodel(inputLayer,classes=len(classes))
        model = self.denseNet(inputLayer,classes=len(self.vertebra_label),inputshape=X.shape[-3:])
        model.compile(optimizer="adam",loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
        # model.summary()
        print("vertebra...model...over")
        checkpoint = ModelCheckpoint(self.saveModel_vertebra + 'vertebra.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        model.fit(X,y,batch_size=32,epochs=100,callbacks=[checkpoint,reduce_lr,early_stopping],validation_data=(Xv,yv),verbose=2)

        self.vertebra_model = model
        pass

    def loadTestvertebra(self):
        if self.vertebra_model == None:
            inputLayer = keras.layers.Input(shape=self.resizeShape+[1])
            model = self.denseNet(inputLayer, classes=len(self.vertebra_label), inputshape= self.resizeShape+[1])
            model.load_weights(r"logs/001/vertebra.h5")
            self.vertebra_model = model
            return model

    def test_vertebra(self,X):
        X = X.reshape([1]+self.resizeShape+[1])
        ypre = self.vertebra_model.predict(X)
        return ypre

    def train_disc_data(self):
        X = np.load(r"{}\imgesSlice_disc_train.npy".format(self.filename),allow_pickle=True)
        X = self.npy2arry(X)

        y = np.load(r"{}\ySlice_disc_train.npy".format(self.filename),allow_pickle=True)

        Xv = np.load(r"{}\imgesSlice_disc_val.npy".format(self.filename),allow_pickle=True)
        Xv = self.npy2arry(Xv)

        yv = np.load(r"{}\ySlice_disc_val.npy".format(self.filename),allow_pickle=True)

        classes = self.disc_label
        y = self.npy2y(y,classes)
        yv = self.npy2y(yv, classes)
        return X,y,Xv,yv

    def train_disc(self):
        X, y, Xv, yv = self.train_disc_data()

        inputLayer = keras.layers.Input(shape=X.shape[-3:])
        # model = self.CNNmodel(inputLayer,classes=len(classes))
        model = self.denseNet(inputLayer, classes=len(self.disc_label), inputshape=X.shape[-3:])
        model.compile(optimizer="adam",loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
        # model.summary()
        print("disc...model...over")
        checkpoint = ModelCheckpoint(self.saveModel_disc + 'disc.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        model.fit(X,y,batch_size=32,epochs=100,callbacks=[checkpoint,reduce_lr,early_stopping],validation_data=(Xv,yv),verbose=2)

        self.disc_model = model
        pass

    def loadTestdisc(self):
        if self.disc_model == None:
            inputLayer = keras.layers.Input(shape=self.resizeShape+[1])
            model = self.denseNet(inputLayer, classes=len(self.disc_label), inputshape=self.resizeShape+ [1])
            model.load_weights(r"logs/002/disc.h5")
            self.disc_model = model
            return model

    def test_disc(self,X):
        X = X.reshape([1]+self.resizeShape+[1])
        ypre = self.disc_model.predict(X)
        return ypre


    def train_discv5_data(self):
        X = np.load(r"{}\imgesSlice_discv5_train.npy".format(self.filename),allow_pickle=True)
        X = self.npy2arry(X)

        y = np.load(r"{}\ySlice_discv5_train.npy".format(self.filename),allow_pickle=True)

        Xv = np.load(r"{}\imgesSlice_discv5_val.npy".format(self.filename),allow_pickle=True)
        Xv = self.npy2arry(Xv)

        yv = np.load(r"{}\ySlice_discv5_val.npy".format(self.filename),allow_pickle=True)

        classes = self.disc_labelv5
        y = self.npy2y(y,classes)
        yv = self.npy2y(yv, classes)
        return X,y,Xv,yv

    def binary_PFA(self,y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # N = total number of negative labels
        N = K.sum(1 - y_true)+ K.epsilon()
        # FP = total number of false alerts, alerts from the negative class labels
        FP = K.sum(y_pred - y_pred * y_true)
        return FP / N

    def binary_PTA(self,y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # P = total number of positive labels
        P = K.sum(y_true) + K.epsilon()
        # TP = total number of correct alerts, alerts from the positive class labels
        TP = K.sum(y_pred * y_true)
        return TP / P

    def auc(self,y_true, y_pred):
        ptas = tf.stack([self.binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
        pfas = tf.stack([self.binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
        pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
        binSizes = -(pfas[1:] - pfas[:-1])
        s = ptas * binSizes
        return K.sum(s, axis=0)

    def focal_loss_fixed(self,y_true, y_pred):
        gamma = 2
        alpha = 0.25
    # tensorflow backend, alpha and gamma are hyper-parameters which can set by you

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


    def train_discv5(self):
        X, y, Xv, yv = self.train_discv5_data()

        inputLayer = keras.layers.Input(shape=X.shape[-3:])

        # model = self.CNNmodel(inputLayer,classes=len(classes))
        # model = self.denseNet(inputLayer, classes=len(self.disc_labelv5), inputshape=X.shape[-3:])
        model = self.ResNet(inputLayer, classes=len(self.disc_labelv5), inputshape=X.shape[-3:])

        model.compile(optimizer="adam",loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_crossentropy'])
        # model.summary()
        print("椎间盘...model...over")
        checkpoint = ModelCheckpoint(self.saveModel_discv5 + 'discv5.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        model.fit(X,y,batch_size=32,epochs=100,callbacks=[checkpoint,reduce_lr,early_stopping],validation_data=(Xv,yv),verbose=2)

        self.disc_model = model
        pass

    def loadTestdiscv5(self):
        if self.disc_model == None:
            inputLayer = keras.layers.Input(shape=self.resizeShape+[1])
            # model = self.denseNet(inputLayer, classes=len(self.disc_labelv5), inputshape=self.resizeShape+ [1])
            model = self.ResNet(inputLayer, classes=len(self.disc_labelv5), inputshape=self.resizeShape + [1])
            model.load_weights(r"logs/004/discv5.h5")
            self.disc_model = model
            return model

    def test_discv5(self,X):
        X = X.reshape([1]+self.resizeShape+[1])
        ypre = self.disc_model.predict(X)
        return ypre

    def clear(self):
        K.clear_session()

def _main():
    cnn = CNN(fileName=r"data2class2",sliceResize=[48, 32])
    cnn.train_discv5()
    cnn.train_vertebra()
    cnn.train_disc() #v1-v4


def ReadySlice2class(dataTxt=r"resultStep2.txt",resultTxt=r"resultStep3.txt",sliceResize =  None):
    '''
    切片来自于dataTxt对应的图像，所以切片时去jpg里面找图来切
    :param dataTxt:
    :param resultTxt:
    :return:
    '''
    f = open(dataTxt,'r')
    f1 = open(resultTxt,'w')
    datas = f.readlines()

    vertebraSlice = {}
    discSlice = {}

    cnn_disc = CNN(sliceResize)
    model_disc = cnn_disc.loadTestdisc()
    cnn_discv5 = CNN(sliceResize)
    model_discv5 = cnn_discv5.loadTestdiscv5()
    cnn_vertebra = CNN(sliceResize)
    model_vertebra = cnn_vertebra.loadTestvertebra()


    prex,prey = 0,0
    for data in datas:
        d = data.split()
        imgPath = d[0]
        img = cv2.imread(imgPath)

        mean = np.mean(img)
        var = np.mean(np.square(img - mean))
        img = (img - mean) / np.sqrt(var)

        m, n, _ = img.shape
        vertebraSlice[imgPath] = []
        discSlice[imgPath] = []
        f1.write(imgPath)

        xy = np.array(list(map(lambda x:list(map(int,x.split(',')[1:])), d[1:]))) #从小到大
        xy = sorted(xy,key=lambda x:x[1])
        deta = []
        for i in range(len(xy)-1):
            deta.append(xy[i+1][1]-xy[i][1])
        deta = np.mean(deta)

        for target in d[1:]:
            target = target.split(',')
            x, y = list(map(int, target[1:]))
            if target[0] == 'disc':
                w1, h1 = m / 5, n / 20  # 椎间盘要求更长而不是更高  n控制高度
                offset = 5
                yoffset = 3
                miny = y - deta
                maxy = y + deta
                minx = x + offset - w1 / 2
                maxx = x + offset + w1 / 2
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
                f1.write(" " + str(x) + "," + str(y) + "," + target[0])
                nimg_x = img[int(miny):int(maxy), int(minx):int(maxx)]

                xlen2 = int((maxx-minx)/2)
                ylen2 = int((maxy - miny) / 2)

                nimg_x_r = nimg_x[:, int(xlen2*2/3):]
                print(imgPath, "---", xlen2, "--img",img.shape, "--nimg_x",nimg_x.shape,"--nimg_x_r",nimg_x_r.shape,"--",deta,"--x",(int(minx),int(maxx)),"--",(int(miny),int(maxy)))
                nimg_x_r = cv2.resize(nimg_x_r, (sliceResize[0], sliceResize[1]), interpolation=cv2.INTER_CUBIC)

                # plt.figure(1)
                # plt.imshow(nimg_x_r)
                # plt.show()

                X = nimg_x_r[:, :, 1].reshape([1] + sliceResize + [1])
                p = model_disc.predict(X)
                #v1-v4
                # pm = np.max(p)
                pre = np.argmax(p, axis=1)

                # v1,v5
                nimg_x = cv2.resize(nimg_x, (sliceResize[0], sliceResize[1]), interpolation=cv2.INTER_CUBIC)

                # 预测
                X = nimg_x[:, :, 1].reshape([1] + sliceResize + [1])
                p = model_discv5.predict(X)
                pre1 = np.argmax(p, axis=1)[0]
                f1.write(","+cnn_disc.disc_label[int(pre)]+","+str(pre1))
                # print("dics 类别(正常,膨出,突出,脱出,椎体内疝出{}):{}".format(cnn_disc.disc_label,cnn_disc.disc_label[int(pre)]))
                prex = x
                prey = y
            else:
                w1, h1 = m / 6, n / 12  # 识别框的宽度和高度 更大
                offset = 5
                yoffset = 3
                miny = y - deta
                maxy = y + deta
                minx = x + offset - w1 / 2
                maxx = x + offset + w1 / 2
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
                f1.write(" " + str(x) + "," + str(y) + "," + target[0])
                nimg_x = img[int(miny):int(maxy), int(minx):int(maxx)]
                nimg_x = cv2.resize(nimg_x, (sliceResize[0], sliceResize[1]))[:, :, 1]

                X = nimg_x.reshape([1] + sliceResize + [1])
                p = model_vertebra.predict(X)

                # p = cnn_vertebra.test_vertebra(nimg_x)
                pm = np.max(p)
                pre = np.argmax(p, axis=1)
                f1.write(","+cnn_vertebra.vertebra_label[int(pre)]+","+str(pm))
                # print("vertebra 类别({}):{}".format(cnn_vertebra.vertebra_label, cnn_vertebra.vertebra_label[int(pre)]))
            # cv2.rectangle(img,(int(x - w1 / 2),int(y + h1 / 2)),(int(x + w1 / 2),int(y - h1 / 2)),color=[0,0,255])
            # cv2.imshow("1", img)
            # cv2.imshow("2", nimg_x)
            # cv2.waitKey(0)
        f1.write("\n")
    f.close()
    f1.close()

if __name__ == "__main__":
    #训练模型
    cnn = CNN()
    # train = True
    # if train:
    #     cnn.train_vertebra()
    #     cnn.train_disc()

    # 对分类数据进行分析
    # X, y, Xv, yv = cnn.train_disc_data() #5 class
    # X, y, Xv, yv = cnn.train_vertebra_data()  # 2 class
    # _main()

