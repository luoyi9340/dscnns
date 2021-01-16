# -*- coding: utf-8 -*-  
'''
训练dscnns降噪网络

@author: luoyi
Created on 2021年1月17日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('dscnns')[0]
ROOT_PATH = ROOT_PATH + "dscnns"
sys.path.append(ROOT_PATH)


import data.dataset as ds
import utils.conf as conf
from models.dscnns import DSCNNsModel


#    准备数据集
db_train = ds.generator_db_tf(in_dir=conf.DATASET.get_in_train(), 
                              count=conf.DATASET.get_count_train(), 
                              label_dir=conf.DATASET.get_label_train(), 
                              batch_size=conf.TRAIN.get_batch_size(),
                              epochs=conf.TRAIN.get_epochs())
db_val = ds.generator_db_tf(in_dir=conf.DATASET.get_in_val(),
                            count=conf.DATASET.get_count_val(),
                            label_dir=conf.DATASET.get_label_val(),
                            batch_size=conf.TRAIN.get_batch_size(),
                            epochs=conf.TRAIN.get_epochs())


#    准备网络
dscnns_model = DSCNNsModel(input_shape=(conf.DATASET.get_image_height(), conf.DATASET.get_image_width(), 3),
                           base_channel_num=conf.DSCNNS.get_base_channel_num(),
                           block_num=conf.DSCNNS.get_block_num(),
                           is_bias=conf.DSCNNS.get_is_bias())
dscnns_model.summary()


#    喂数据
batch_size=conf.TRAIN.get_batch_size()
epochs=conf.TRAIN.get_epochs()
steps_per_epoch=conf.DATASET.get_count_train() / batch_size
his = dscnns_model.fit_generator(generator=db_train, 
                           validation_data=db_val,
                           steps_per_epoch=steps_per_epoch, 
                           epochs=epochs, 
                           verbose=1, 
                           callbacks=dscnns_model.get_callbacks())
print(his)
