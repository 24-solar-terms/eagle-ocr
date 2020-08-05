# coding:utf-8
import sys
import os
sys.path.insert(1, "./crnn")
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from . import util
from . import dataset
from .models import crnn as crnn
from . import keys_crnn
from math import *
import cv2

GPU = True


def dumpRotateImage_(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    height, width = imgOut.shape[:2]
    return imgOut


def crnnSource():
    alphabet = keys_crnn.alphabet
    converter = util.strLabelConverter(alphabet)
    if torch.cuda.is_available() and GPU:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
    else:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cpu()
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'samples/model_acc97.pth')
    model.eval()
    model.load_state_dict(torch.load(path))
    return model, converter


##加载模型
model, converter = crnnSource()

# 我的注释31：此处为根目录模型使用pytorch框架实现的识别方法
def crnnOcr(image):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im
    @@text_recs:text box

    """
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    # 我的注释32：上面操作的意思是把height变成32，相应的width应该是多少，即w，将图片resize成w x 32的大小
    # print "im size:{},{}".format(image.size,w)
    
    transformer = dataset.resizeNormalize((w, 32))
    # 我的注释32：返回来一个resize转换器，转换的规定尺寸是w x 32
    # 根据源码大致是根据双线性插值法resize到规定大小，然后进行相关缩放操作后返回一个tensor对象
    # 内使用ToTensor方法把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，
    # 取值范围是[0,1.0]的torch.FloadTensor
    if torch.cuda.is_available() and GPU:
        image = transformer(image).cuda()
    else:
        image = transformer(image).cpu()
    
    # 我的注释32：通过上面操作可以看出将图片变成了一通道高度32的灰度图

    # 我的注释32：transformer方法返回来的是tensor对象，view方法保证数据不变，有点类似reshape操作，打印发现
    # image.size返回结果是torch.Size([1, 32, 205])，应该是[C,H,W]的形式
    image = image.view(1, *image.size())
    # 我的注释32：使用view方法后每个图片的size变成torch.Size([1, 1, 32, 171])多了一维，不知为何
    
    image = Variable(image)
    # 我的注释32：用Variable包含tensor，可对任意标量函数进行求导
    
    # 我的注释33：网上查了下 model.eval() ：不启用 BatchNormalization 和 Dropout
    model.eval()
    
    # 我的注释34：获得预测结果
    preds = model(image)
    
    
    _, preds = preds.max(2)
    
    preds = preds.transpose(1, 0).contiguous().view(-1)
    
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    # 我的注释35：又对原始预测结果进行了一些神奇的转换
    
    # 我的注释36：解码过程
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    if len(sim_pred) > 0:
        if sim_pred[0] == u'-':
            sim_pred = sim_pred[1:]

    return sim_pred
