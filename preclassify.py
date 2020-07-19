import os
import numpy as np
import SimpleITK as sitk
import re
import json
import cv2

import keras
from yolo3 import densenet
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K

class Gen(keras.utils.Sequence):
    """
    'Generates data for Keras'

    list_IDs - list of files that this generator should load
    labels - dictionary of corresponding (integer) category to each file in list_IDs

    Expects list_IDs and labels to be of the same length
    """

    def __init__(self, X, Y = None,batch_size=64, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = Y
        self.X = list(X)

        self.shuffle = shuffle
        self.on_epoch_end()

        self.shapeResize = [256,256]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Generate data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        X_list = [self.X[k] for k in indexes]
        y_list = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(X_list,y_list)
        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def dicom2array(self,dcm_path):
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

    def __data_generation(self, x,y):
        #'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, self.shapeResize[0],self.shapeResize[1],1 ))
        Y = np.empty((self.batch_size))

        # Generate data
        for i, curX in enumerate(x):
            img = self.dicom2array(curX)
            img = cv2.resize(img,(self.shapeResize[0],self.shapeResize[1])).reshape(self.shapeResize+[1])
            X[i] = img
            Y[i] = y[i]
        return X, Y

class CNN(object):
    def __init__(self):
        self.saveModel_pre = r"logs/003/"

        self.pre_model = None

        self.resizeShape = [256,256]
        pass

    def denseNet(self,input,classes,inputshape):
        return densenet.DenseNet(img_input=input,blocks=[2,3,2,0],classes=classes,input_shape=inputshape)

    def train_T(self):
        f = open("preclassifyTrain.json",'r')
        data = json.load(f)
        f.close()

        trainFile = data["files"]
        trainY = data["targets"]

        genTrain = Gen(X=trainFile,Y=trainY,batch_size=32)
        X,y = genTrain.__getitem__(0)
        print(X.shape,y.shape)

        f = open("preclassifyVal.json",'r')
        data = json.load(f)
        f.close()

        valFile = data["files"]
        valY = data["targets"]

        genVal = Gen(X=valFile,Y=valY,batch_size=32)

        inputLayer = keras.layers.Input(shape = self.resizeShape + [1])

        model = self.denseNet(inputLayer,classes=3, inputshape = self.resizeShape + [1])
        model.compile(optimizer="adam",loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
        # model.summary()

        checkpoint = ModelCheckpoint(self.saveModel_pre + 'preClassify.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        model.fit_generator(generator=genTrain,validation_data=genVal,epochs=30,callbacks=[checkpoint,reduce_lr,early_stopping],verbose=2)

        self.pre_model = model
        pass

    def loadModel(self):
        if self.pre_model == None:
            inputLayer = keras.layers.Input(shape=self.resizeShape+[1])
            model = self.denseNet(inputLayer, classes=3, inputshape= self.resizeShape+[1])
            model.load_weights(r"logs/003/preClassify.h5")
            self.pre_model = model
            return model

    def test_(self,X):
        X = X.reshape([1]+self.resizeShape+[1])
        ypre = self.pre_model.predict(X)
        return ypre


    def clear(self):
        K.clear_session()

def dicom2array1(dcm_path):
    '''
    target: t2sag:0,t1sag:1,t2tra:2
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    try:
        image_file_reader.ReadImageInformation()
    except:
        return None

    image = image_file_reader.Execute()
    target = None
    if image.GetNumberOfComponentsPerPixel() == 1:
        try:
            description = image_file_reader.GetMetaData('0008|103e')
            inde = int(image_file_reader.GetMetaData('0020|0013'))
            if inde >=2 and inde <=7:
                if re.match(r"(.*)[tT]2(.*)[sS][aA][gG](.*)",description,0) or\
                        re.match(r"(.*)[sS][aA][gG](.*)[tT]2(.*)",description,0):
                     #判断是否为T2
                    target = 0
                elif re.match(r"(.*)T1(.*)SAG", description):
                    target = 1
                elif re.match(r"(.*)T2(.*)TRA",description):
                    target = 2
                else:
                    target = None
            else:
                target = None
        except:
            target = None
    return target

def readyPreclssify(dataPath):
    '''
    准备josn数据，里面包含每个study的文件和对应标签
    :return:
    '''

    study = os.listdir(dataPath)
    temp = {"files": [],
            "targets": []}
    for studyI in study:
        Impath = os.listdir(os.path.join(dataPath, studyI))


        print("study:", studyI)

        for Impathi in Impath:
            target = dicom2array1(os.path.join(os.path.join(dataPath, studyI), Impathi))
            if target != None:
                temp["files"].append(os.path.join(os.path.join(dataPath, studyI), Impathi))
                temp["targets"].append(target)

    f1 = open("preClassify.json",'w')
    json.dump(temp,f1)
    f1.close()

    np.random.seed(2020)
    all_ = np.arange(0, len(temp["files"]))
    valIndex = np.random.randint(low=0, high=len(temp["files"]), size=int(len(temp["files"]) * 0.1))
    trainIndex = [i for i in all_ if i not in valIndex]

    tempTrain = {"files": [],
                "targets": []}
    f = open("preclassifyTrain.json",'w')
    for train in trainIndex:
        tempTrain["files"].append(temp["files"][train])
        tempTrain["targets"].append(temp["targets"][train])
    json.dump(tempTrain,f)
    f.close()

    tempVal = {"files": [],
                "targets": []}
    f = open("preclassifyVal.json",'w')
    for val in valIndex:
        tempVal["files"].append(temp["files"][val])
        tempVal["targets"].append(temp["targets"][val])
    json.dump(tempVal,f)
    f.close()

    print("over...")

    pass

def test():
    print("over...")


if __name__ == "__main__":
    dataPath = r"H:\dataBase\tianchi_spinal\lumbar_train51\train"
    readyPreclssify(dataPath=dataPath)
    cnn = CNN()
    cnn.train_T()

