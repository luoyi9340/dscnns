# -*- coding: utf-8 -*-  
'''
DSCNNs网络

@author: luoyi
Created on 2021年1月16日
'''
import tensorflow as tf

import utils.conf as conf

#    dscnns网络
class DSCNNsModel(tf.keras.Model):
    '''dscnns网络结构
        ------------------------------ layer in ------------------------------
        conv:[3*3*64] strides=1 padding=same out=[180 * 480 * base_channel_num]
        ------------------------------ layer block ------------------------------
        conv:[3*3*64] strides=1 padding=same out=[180 * 480 * base_channel_num]
        bn:
        relu:
        time: block_num（该层循环block_num次）
        ------------------------------ layer out ------------------------------
        conv:[3*3*64] strides=1 padding=same out=[180 * 480 * base_channel_num]
    '''
    def __init__(self, 
                 name='DSCNNsNet', 
                 input_shape=(conf.DATASET.get_image_height(), conf.DATASET.get_image_width(), 3), 
                 base_channel_num=64, 
                 block_num=16, 
                 is_bias=False, 
                 **kwargs):
        super(DSCNNsModel, self).__init__(name=name, **kwargs)
        
        self.__input_shape = input_shape
        self.__base_channel_num = base_channel_num
        self.__block_num = block_num
        self.__is_bias = is_bias
        
        #    装配网络
        self.__assembling(base_channel_num, block_num, is_bias)
        #    各种回调
        self.__registory_callbacks()
        #    编译网络
        self.__compile_model()
        
        self.build(input_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
        pass
    
    #    装配网络
    def __assembling(self, base_channel_num, block_num, is_bias):
        kernel_initializer = tf.initializers.he_normal()
        bias_initializer = tf.initializers.zeros()
        
        #    layer in
        self.layer_in = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=base_channel_num, kernel_size=[3, 3], strides=1, padding='same', input_shape=self.__input_shape, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, use_bias=is_bias)
            ], name='layer_in')
        
        block_shape = (self.__input_shape[0], self.__input_shape[1], base_channel_num)
        #    layer block
        self.layer_block = tf.keras.models.Sequential(name='layer_block')
        for i in range(block_num):
            self.layer_block.add(tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(name='layer_block_conv_' + str(i), filters=base_channel_num, kernel_size=[3, 3], strides=1, padding='same', input_shape=block_shape, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, use_bias=is_bias),
                tf.keras.layers.BatchNormalization(name='layer_block_bn_' + str(i)),
                tf.keras.layers.ReLU(name='layer_block_relu_' + str(i))
                ], name='layer_block_' + str(i)))
            pass
        
        #    layer out
        self.layer_out = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=self.__input_shape[-1], kernel_size=[3, 3], strides=1, padding='same', input_shape=block_shape, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, use_bias=is_bias)
            ], name='layer_out')
        pass
    
    #    注册各种回调
    def __registory_callbacks(self):
        callbacks = []
        #    动态lr
        reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                    factor=0.1,             #    每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                                                    patience=1,             #    当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                                                                    mode='auto',            #    ‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少
                                                                    epsilon=0.00001,        #    阈值，用来确定是否进入检测值的“平原区” 
                                                                    cooldown=0,             #    学习率减少后，会经过cooldown个epoch才重新进行正常操作
                                                                    min_lr=0                #    学习率的下限（下不封顶）
                                                                    )
        callbacks.append(reduce_lr_on_plateau)
        #    tensorboard
        tensorboard_dir = conf.TRAIN.get_tensorboard_out() + "/" + self.name + "_b" + str(conf.TRAIN.get_batch_size()) + "_lr" + str(conf.TRAIN.get_learning_rate())
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir,               #    tensorboard主目录
                                                         histogram_freq=1,                      #    对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。 
                                                                                                #        如果设置成 0 ，直方图不会被计算。对于直方图可视化的验证数据（或分离数据）一定要明确的指出。
                                                         write_graph=True,                      #    是否在 TensorBoard 中可视化图像。 如果 write_graph 被设置为 True
                                                         write_grads=True,                      #    是否在 TensorBoard 中可视化梯度值直方图。 
                                                                                                #        histogram_freq 必须要大于 0
                                                         batch_size=conf.TRAIN.get_batch_size(),                 #    用以直方图计算的传入神经元网络输入批的大小
                                                         write_images=True,                     #    是否在 TensorBoard 中将模型权重以图片可视化，如果设置为True，日志文件会变得非常大
                                                         embeddings_freq=None,                  #    被选中的嵌入层会被保存的频率（在训练轮中）
                                                         embeddings_layer_names=None,           #    一个列表，会被监测层的名字。 如果是 None 或空列表，那么所有的嵌入层都会被监测。
                                                         embeddings_metadata=None,              #    一个字典，对应层的名字到保存有这个嵌入层元数据文件的名字
                                                         embeddings_data=None,                  #    要嵌入在 embeddings_layer_names 指定的层的数据。 Numpy 数组（如果模型有单个输入）或 Numpy 数组列表（如果模型有多个输入）
                                                         update_freq='batch'                    #    'batch' 或 'epoch' 或 整数。
                                                                                                #        当使用 'batch' 时，在每个 batch 之后将损失和评估值写入到 TensorBoard 中。
                                                                                                #        同样的情况应用到 'epoch' 中。
                                                                                                #        如果使用整数，例如 10000，这个回调会在每 10000 个样本之后将损失和评估值写入到 TensorBoard 中。注意，频繁地写入到 TensorBoard 会减缓你的训练。
                                                         )
        callbacks.append(tensorboard)
        #    每轮epoch保存模型
        auto_save_file_path = conf.TRAIN.get_save_weights_out() + "/" + self.name + ".h5"
        conf.mkfiledir_ifnot_exises(auto_save_file_path)
        auto_save_weights_callback = tf.keras.callbacks.ModelCheckpoint(filepath=auto_save_file_path,
                                                                            monitor="val_loss",         #    需要监视的值
                                                                            verbose=1,                      #    信息展示模式，0或1
                                                                            save_best_only=True,            #    当设置为True时，将只保存在验证集上性能最好的模型，一般我们都会设置为True. 
                                                                            model='auto',                   #    ‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
                                                                                                            #    例如:
                                                                                                            #        当监测值为val_acc时，模式应为max，
                                                                                                            #        当检测值为val_loss时，模式应为min。
                                                                                                            #        在auto模式下，评价准则由被监测值的名字自动推断。 
                                                                            save_weights_only=True,         #    若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等
                                                                            period=1                        #    CheckPoint之间的间隔的epoch数
                                                                            )
        callbacks.append(auto_save_weights_callback)
        self.__callbacks = callbacks
        pass
    #    取回调
    def get_callbacks(self):
        return self.__callbacks
    
    #    编译网络
    def __compile_model(self):
        self.__loss = tf.losses.mse
        self.__optimizer = tf.optimizers.Adam(learning_rate=conf.TRAIN.get_learning_rate())
        self.__metric = [tf.metrics.mse]
        
        self.compile(optimizer=self.__optimizer, 
                     loss=self.__loss, 
                     metrics=self.__metric)
        pass
    
    #    前向传播
    def call(self, x, training=None, mask=None):
        y = self.layer_in.call(x, training=training, mask=mask)
        y = self.layer_block.call(y, training=training, mask=mask)
        y = self.layer_out.call(y, training=training, mask=mask)
        #    输出噪声
        return x - y
    pass

