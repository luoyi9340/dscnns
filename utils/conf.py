# -*- coding: utf-8 -*-  
'''
Created on 2020年12月15日

@author: irenebritney
'''
import yaml
import os
import sys


#    取项目根目录（其他一切相对目录在此基础上拼接）
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('dscnns')[0]
ROOT_PATH = ROOT_PATH + "dscnns"


#    取配置文件目录
CONF_PATH = ROOT_PATH + "/resources/conf.yml"
#    加载conf.yml配置文件
def load_conf_yaml(yaml_path=CONF_PATH):
    print('加载配置文件:' + CONF_PATH)
    f = open(yaml_path, 'r', encoding='utf-8')
    fr = f.read()
    
#     c = yaml.load(fr, Loader=yaml.SafeLoader)
    c = yaml.safe_load(fr)
    
    #    读取letter相关配置项
    dataset = Dataset(c['dataset']['in_train'], c['dataset']['label_train'], c['dataset']['count_train'],
                      c['dataset']['in_val'], c['dataset']['label_val'], c['dataset']['count_val'],
                      c['dataset']['in_test'], c['dataset']['label_test'], c['dataset']['count_test'],
                      c['dataset']['image_width'], c['dataset']['image_height'])
    
    dscnns = DSCNNs(c['dscnns']['base_channel_num'],
                    c['dscnns']['block_num'],
                    c['dscnns']['is_bias'])
    
    train = Train(c['train']['batch_size'],
                  c['train']['epochs'],
                  c['train']['tensorboard_out'],
                  c['train']['learning_rate'],
                  c['train']['save_weights_out'])
    
    return c, dataset, dscnns, train


#    验证码识别数据集。为了与Java的风格保持一致
class Dataset:
    def __init__(self, 
                 in_train="", label_train='', count_train=0,
                 in_val="", label_val="", count_val=0,
                 in_test="", label_test="", count_test=0,
                 image_width=480, image_height=180):
        self.__in_train = convert_to_abspath(in_train)
        self.__label_train = convert_to_abspath(label_train)
        self.__count_train = count_train
        
        self.__in_val = convert_to_abspath(in_val)
        self.__label_val = convert_to_abspath(label_val)
        self.__count_val = count_val
        
        self.__in_test = convert_to_abspath(in_test)
        self.__label_test = convert_to_abspath(label_test)
        self.__count_test = count_test
        
        self.__image_width = image_width
        self.__image_height = image_height
        pass
    def get_in_train(self): return convert_to_abspath(self.__in_train)
    def get_count_train(self): return self.__count_train
    def get_label_train(self): return convert_to_abspath(self.__label_train)
    
    def get_in_val(self): return convert_to_abspath(self.__in_val)
    def get_count_val(self): return self.__count_val
    def get_label_val(self): return convert_to_abspath(self.__label_val)
    
    def get_in_test(self): return convert_to_abspath(self.__in_test)
    def get_count_test(self): return self.__count_test
    def get_label_test(self): return convert_to_abspath(self.__label_test)
    
    def get_image_width(self): return self.__image_width
    def get_image_height(self): return self.__image_height
    pass


#    dscnns网络相关配置
class DSCNNs():
    def __init__(self, base_channel_num=64, block_num=16, is_bias=False):
        self.__base_channel_num = base_channel_num
        self.__block_num = block_num
        self.__is_bias = is_bias
        pass
    def get_base_channel_num(self): return self.__base_channel_num
    def get_block_num(self): return self.__block_num
    def get_is_bias(self): return self.__is_bias
    pass


#    train相关配置
class Train():
    def __init__(self, batch_size=32, epochs=5, tensorboard_out='logs/tensorboard', learning_rate=0.01, save_weights_out='temp/models'):
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__tensorboard_out = tensorboard_out
        self.__learning_rate = learning_rate
        self.__save_weights_out = save_weights_out
        pass
    def get_batch_size(self): return self.__batch_size
    def get_epochs(self): return self.__epochs
    def get_tensorboard_out(self): return convert_to_abspath(self.__tensorboard_out)
    def get_learning_rate(self): return self.__learning_rate
    def get_save_weights_out(self): return convert_to_abspath(self.__save_weights_out)
    pass


#    取配置的绝对目录
def convert_to_abspath(path):
    '''取配置的绝对目录
        "/"开头的目录原样输出
        非"/"开头的目录开头追加项目根目录
    '''
    if (path.startswith("/")):
        return path
    else:
        return ROOT_PATH + "/" + path
    
#    检测文件所在上级目录是否存在，不存在则创建
def mkfiledir_ifnot_exises(filepath):
    '''检测log所在上级目录是否存在，不存在则创建
        @param filepath: 文件目录
    '''
    _dir = os.path.dirname(filepath)
    if (not os.path.exists(_dir)):
        os.makedirs(_dir)
    pass
#    检测目录是否存在，不存在则创建
def mkdir_ifnot_exises(_dir):
    '''检测log所在上级目录是否存在，不存在则创建
        @param dir: 目录
    '''
    if (not os.path.exists(_dir)):
        os.makedirs(_dir)
    pass

ALL_DICT, DATASET, DSCNNS, TRAIN = load_conf_yaml()


#    写入配置文件
def write_conf(_dict, file_path):
    '''写入当前配置项的配置文件
        @param dict: 要写入的配置项字典
        @param file_path: 文件path
    '''
    file_path = convert_to_abspath(file_path)
    mkfiledir_ifnot_exises(file_path)
    
    #    存在同名文件先删除
    if (os.path.exists(file_path)):
        os.remove(file_path)
        pass
    
    fw = open(file_path, mode='w', encoding='utf-8')
    yaml.safe_dump(_dict, fw)
    fw.close()
    pass


#    追加sys.path
def append_sys_path(path):
    path = convert_to_abspath(path)
    sys.path.append(path)
    print(sys.path)
    pass
