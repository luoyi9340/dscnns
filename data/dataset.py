# -*- coding: utf-8 -*-  
'''
DSCNNs数据集


@author: luoyi
Created on 2021年1月17日
'''
import tensorflow as tf
import os
import PIL
import numpy as np


import utils.conf as conf


#    文件迭代器
def file_generator(in_dir='', count=10, label_dir='', 
                   x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                   y_preprocess=lambda y:((y / 255.) - 0.5) * 2):
    #    理论上两个目录下的文件名应该都一样。遍历一label目录，用遍历到的文件名从另一个中取
    flist = os.listdir(label_dir)
    i = 0
    for fname in flist:
        if (i >= count): break;
        i += 1
        
        train_fpath = in_dir + '/' + fname
        label_fpath = label_dir + '/' + fname
        if (not os.path.exists(train_fpath)): continue
        if (not os.path.isfile(train_fpath)): continue
        if (not os.path.isfile(label_fpath)): continue
        
        #    读取图片信息，并且归一化
        x_image = PIL.Image.open(train_fpath, mode='r')
        x_image = x_image.resize((conf.DATASET.get_image_width(), conf.DATASET.get_image_height()), PIL.Image.ANTIALIAS)
        x = np.asarray(x_image, dtype=np.float32)
        if (x_preprocess): x = x_preprocess(x)
        
        y_image = PIL.Image.open(label_fpath, mode='r')
        y_image = y_image.resize((conf.DATASET.get_image_width(), conf.DATASET.get_image_height()), PIL.Image.ANTIALIAS)
        y = np.asarray(y_image, dtype=np.float32)
        if (y_preprocess): y = y_preprocess(y)
        
        
        #    实际的y数据是 原图 - 无噪图
        yield x, y
        pass
    pass

#    迭代器数据集
def generator_db_tf(in_dir='', count=0, label_dir='', batch_size=32, epochs=5, x_preprocess=None, y_preprocess=None):
    '''迭代器数据集
        @param in_dir: 输入文件目录
        @param count: 读多少条数据
        @param label_dir: 标签文件目录
        @param batch_size: 批量大小
        @param x_preprocess: x数据前置处理
        @param y_preprocess: y数据前置处理
        @return: dataset.from_generator
    '''
    x_shape = tf.TensorShape([conf.DATASET.get_image_height(), conf.DATASET.get_image_width(), 3])
    y_shape = tf.TensorShape([conf.DATASET.get_image_height(), conf.DATASET.get_image_width(), 3])
    db = tf.data.Dataset.from_generator(generator=lambda :file_generator(in_dir=in_dir, 
                                                                    label_dir=label_dir,
                                                                    count=count,
                                                                    x_preprocess=x_preprocess,
                                                                    y_preprocess=y_preprocess), 
                                        output_types=(tf.float32, tf.float32), 
                                        output_shapes=(x_shape, y_shape))
    db = db.shuffle(buffer_size=batch_size * 128)
    db = db.batch(batch_size)
    db = db.repeat(epochs)
    return db