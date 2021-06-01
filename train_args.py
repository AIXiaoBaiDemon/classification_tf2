# Author  : WangXiao
# File    : train_configs.py
# Function: 设置模型训练的参数
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import os



class Train_Args(object):
    def __init__(self):

        '''初始化数据信息'''
        self.dataset = 'test_data'  # 设置数据路径
        self.tfrecord_dir = 'tfrecord'   # 设置生成的tfrecord的文件夹地址

        self.train_p = 0.4
        self.val_p = 0.3
        self.image_type = ['jpg','JPG','bmp','png']     # 数据中包含的图片格式
        self.is_tfrecord_generate = True    # 是否生成tfrecords文件，如果已经生成则为False


        """设置模型信息"""
        self.model_name = 'resnet18_cbam'    #设置想要使用的模型，目前提供['densenet'"efficientnet""mobileNet""resnet18 ""resnet18_cbam"]
        self.input_shape = (300,300,3)      # 模型的输入shape
        self.is_load_weight = True      # 是否加载与训练模型
        self.weight_path = 'logs/resnet18_cbam/epoch-004-loss-4.55766-acc-0.535-val_loss-4.54500-val_acc-0.538.h5'  # 与训练模型的地址


        self.optimizer = Adam()     # 优化器选择
        self.epoch = 1000   # 模型迭代次数
        self.batch_size = 32    # 模型的baitchsize
        self.log_dir = 'logs'   # log的存放地址
        self.loss = categorical_crossentropy    # 使用loss
        self.metrics = 'accuracy'   # 使用metrics
        self.learning_rate_base = 1e-4  # 基础lr
        self.is_cosine_scheduler = True     # 是否使用warmup和余弦增强

        self.finetune = True      # 是否进行微调（如果微调则冻结global average pooling之前的层）

