# coding=gbk
from PIL import Image
import numpy as np
# import scipy


def loadImage():
    # ��ȡͼƬ
    im = Image.open("D:\\DataForMining\\pic\\1test.jpg")

    # ��ʾͼƬ
 #   im.show()
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data)
    print(data)
    # �任��512*512
    data = np.reshape(data, (110, 110))
    print(data)
    new_im = Image.fromarray(data)
    # ��ʾͼƬ
#    new_im.show()

loadImage()
