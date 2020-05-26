# -*- coding: utf-8 -*-
## 修复K.ctc_decode bug 当大量测试时将GPU显存消耗完，导致错误，用decode 替代
###
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
sys.path.append('/home/guyu.gy/eagle-ocr/eagle_ocr_model/ocr')
# from PIL import Image
import keras.backend as K
from . import keys_ocr
import numpy as np
from keras.layers import Flatten, BatchNormalization, Permute, TimeDistributed, Dense, Bidirectional, GRU
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import SGD
import tensorflow as tf


config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)
K.set_session(session)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(height, nclass):
    rnnunit = 256
    # 我的注释20：设置CNN 使用CRNN原论文中的CNN结构
    input = Input(shape=(height, None, 1), name='the_input')
    m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(m)

    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
    m = BatchNormalization(axis=1)(m)
    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)
    m = BatchNormalization(axis=1)(m)
    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(m)
    m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)
    # m的输出维度为HWC?
    # 将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层
    # 将维度转成WHC
    
    # 我的注释21：应该是设置RNN
    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='timedistrib')(m)
    m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm1')(m)
    m = Dense(rnnunit, name='blstm1_out', activation='linear')(m)
    m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm2')(m)
    y_pred = Dense(nclass, name='blstm2_out', activation='softmax')(m)
    # 我的注释22：获得基本的模型
    basemodel = Model(inputs=input, outputs=y_pred)
    
    # 我的注释23：一些后续完善，设计keras框架和CRNN原理，后面这这些貌似和model有关，但是后面用来预测的模型是用的basemodel而不是model
    
    labels = Input(name='the_labels', shape=[None, ], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    # model.summary()
    # 我的注释24：所以这个方法应该是得到了一个完整且空白的CRNN网络，使用keras框架搭建起来的。
    return model, basemodel


characters = keys_ocr.alphabet[:]

# 我的注释25：加载预先训练好的CRNN网络的权重W的文件，modelPath指定存放路径

# modelPath = os.path.join(os.getcwd(), "ocr/ocr0.2.h5")
modelPath = '/home/guyu.gy/eagle-ocr/eagle_ocr_model/ocr/ocr0.2.h5'
height = 32

# 我的注释26：应该表示的是能够预测出的汉族种类数，多少种汉字，汉在存在keys_ocr文件中
nclass=len(characters)+1
if os.path.exists(modelPath):
    model, basemodel = get_model(height, nclass)
    basemodel.load_weights(modelPath)
    # basemodel.predict(np.array([np.zeros((32, 32, 1))]))
    # model.load_weights(modelPath)


# 我的注释16：keras模型预测识别文字接口

def predict(im):
    """
    输入图片，输出keras模型的识别结果
    """
    # 我的注释17：按照1：32比例将图片尺寸缩小了
    im = im.convert('L')
    scale = im.size[1] * 1.0 / 32
    w = im.size[0] / scale
    w = int(w)
    im = im.resize((w, 32))
    # 我的注释18：使图像像素取之0-1，应该是便于处理
    img = np.array(im).astype(np.float32) / 255.0
    X = img.reshape((32, w, 1))
    X = np.array([X])
    
    # 我的注释19：以上仍为图像的与处理，basemodel为识别模型，查看basemdel模块和其predict方法
    
    # 我的注释27：使用basemodel进行预测，接口和返回的类型应该是由kears提供
    with session.as_default():
        with session.graph.as_default():
            y_pred = basemodel.predict(X)
    
    y_pred = y_pred[:, 2:, :]
    
    # 我的注释28：根据模型返回的识别结果，可能是5000多个种类的概率数组，然后根据分类结果转成汉字
    # 如果需要彻底搞懂转成汉字的解码过程，首先需要知道返回的结果的形状，然后argmax方法的处理过程
    
    out = decode(y_pred)  ##
    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :]

    # out = u''.join([characters[x] for x in out[0]])

    # 我的注释29：经过一些处理后返回汉字结果
    if len(out) > 0:
        while out[0] == u'。':
            if len(out) > 1:
                out = out[1:]
            else:
                break

    return out


def decode(pred):
    charactersS = characters + u' '
    t = pred.argmax(axis=2)[0]
    # 我的注释30：t数组的确就是预测概率最大的对应文字的下标
    # 根据查看输出结果和对argmax(axis=2)的理解，就是输出结果的每个类别对应了概率，选最大的那个索引作为最终类别
    # 而索引在原类别字符串中恰好对应了那个字，就可以转成汉语了。
    length = len(t)
    char_list = []
    n = len(characters)
    for i in range(length):
        if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(charactersS[t[i]])
    return u''.join(char_list)
