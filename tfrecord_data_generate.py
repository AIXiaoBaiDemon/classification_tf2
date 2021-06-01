# Author  : WangXiao
# File    : make_tfrecord.py
# Function: 输入一个文件夹地址，输出三个tfrecord文件，train，val，test
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import random
import pandas as pd
import numpy as np
import cv2




class TFrecordDataGenerate(object):
    def __init__(self,data_dir,tfrecord_dir,save_shape,is_generate = True,train_p=0.8,val_p=0.1,img_type=['jpg','jpeg','bmp','png']):
        """
        :param data_dir:原始数据文件夹，需要分好明确的类别
        :param tfrecord_dir: tfrecord保存的文件夹路径
        :param save_shape: 图片保存的大小
        :param train_p: 用来train的图片占比
        :param val_p: 用来validation的图片占比
        :param img_type: 原始数据包含的图片格式，jpg or jpeg or bmp or png or others
        """
        self.data_dir = data_dir

        self.tfrecord_dir = tfrecord_dir
        self.save_shape = save_shape
        self.train_p = train_p
        self.val_p = val_p
        self.test_p = 1-self.train_p-self.val_p
        self.img_type = img_type
        self.train_tfrecord_path = os.path.join(self.tfrecord_dir, 'train.tfrecords')
        self.val_tfrecord_path = os.path.join(self.tfrecord_dir, 'val.tfrecords')
        self.test_tfrecord_path = os.path.join(self.tfrecord_dir,'test.tfrecords')
        if is_generate:
            self.write()
        if os.path.exists(os.path.join(tfrecord_dir,'classes.csv')):
            self.classes = np.loadtxt(os.path.join(tfrecord_dir,'classes.csv'),dtype='str')

        self.total_num = self.get_total()


    def get_total(self):
        classes_dir_lsit = [os.path.join(self.data_dir, cls) for cls in self.classes]
        total_num = 0
        for cls_id, cls_dir in enumerate(classes_dir_lsit):
            image_names = [name for name in os.listdir(cls_dir) if name.endswith(tuple(self.img_type))]
            class_total_num = len(image_names)
            total_num+=class_total_num
        return total_num



    def train_val_test_split(self):
        """
        用来进行数据集划分，每个类别按照相应比例划分为train，val，test
        :return: train、val、test的图片路径和对应的label
        """
        # 获取类别
        self.classes = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        print(self.classes)
        classes_dir_lsit = [os.path.join(self.data_dir, cls) for cls in self.classes]

        """获取train、val、test的图片路径和对应label"""
        train_list = []
        val_list = []
        test_list = []
        for cls_id, cls_dir in enumerate(classes_dir_lsit):
            image_names = [name for name in os.listdir(cls_dir) if name.endswith(tuple(self.img_type))]

            class_total_num = len(image_names)
            class_train_num = int(class_total_num * self.train_p)
            class_val_num = int(class_total_num * self.val_p)
            class_test_num = class_total_num - class_train_num - class_val_num
            random.shuffle(image_names)

            train_list.extend([(os.path.join(cls_dir, name), cls_id) for name in image_names[:class_train_num]])
            val_list.extend(
                [(os.path.join(cls_dir, name), cls_id) for name in image_names[class_train_num:class_train_num + class_val_num]])
            test_list.extend([(os.path.join(cls_dir, name), cls_id) for name in image_names[class_train_num + class_val_num:]])
        print("train总数为：{},val总数为：{},test总数为:{}".format(len(train_list), len(val_list), len(test_list)))
        self.num_train = len(train_list)
        self.num_val = len(val_list)
        self.num_test = len(test_list)
        return train_list, val_list, test_list

    def write(self):
        """将图片和label写入tfrecord"""
        # tfrecord_dir = './data_tfrecord'
        if not os.path.exists(self.tfrecord_dir):
            os.mkdir(self.tfrecord_dir)
        # 获取train、val、test的图片路径和对应label
        train_list,val_list,test_list = self.train_val_test_split()
        pd.DataFrame(np.array(train_list)[:, 0]).to_csv(os.path.join(self.tfrecord_dir, 'train.csv'), header=0, index=0)
        pd.DataFrame(np.array(val_list)[:, 0]).to_csv(os.path.join(self.tfrecord_dir, 'val.csv'), header=0, index=0)
        pd.DataFrame(np.array(test_list)[:, 0]).to_csv(os.path.join(self.tfrecord_dir, 'test.csv'), header=0, index=0)
        pd.DataFrame(np.array(self.classes)).to_csv(os.path.join(self.tfrecord_dir, 'classes.csv'), header=0, index=0)

        # target_shape = (300, 300)
        """将图片写入到tfrecord"""
        for set in ['train', 'val', 'test']:
            writer = tf.io.TFRecordWriter(os.path.join(self.tfrecord_dir, set + '.tfrecords'))
            if set == 'train':
                data_list = train_list
                self.train_tfrecord_path = os.path.join(self.tfrecord_dir, set + '.tfrecords')
            elif set == 'val':
                data_list = val_list
                self.val_tfrecord_path = os.path.join(self.tfrecord_dir, set + '.tfrecords')
            else:
                data_list = test_list
                self.test_tfrecord_path = os.path.join(self.tfrecord_dir, set + '.tfrecords')

            for image_path, cls_id in data_list:
                img = cv2.imread(image_path)
                img = cv2.resize(img, self.save_shape)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.astype(np.uint8)
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "image_row": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[cls_id]))
                }))
                writer.write(example.SerializeToString())
            writer.close()
            print(set, ' TFRecord save success')

    def get_train_data(self,batch_size = 32):
        return self._load_tfrecord(batch_size,self.train_tfrecord_path)

    def get_val_data(self,batch_size=32):
        return self._load_tfrecord(batch_size,self.val_tfrecord_path,mode='val')

    def get_test_data(self,batch_size=32):
        return self._load_tfrecord(batch_size,self.test_tfrecord_path,mode='test')

    def _load_tfrecord(self, batch_size:int,preload: str, mode='train'):
        assert os.path.exists(preload)
        inputs = tf.data.TFRecordDataset(preload)
        inputs = inputs.shuffle(buffer_size=2000)
        if mode == 'train':
            inputs = inputs.map(self._parse_train_func, num_parallel_calls=8)
        else:
            inputs = inputs.map(self._parse_eval_func, num_parallel_calls=8)
        inputs = inputs.batch(batch_size)
        return inputs

    def _parse_train_func(self, example_proto):
        """
        train数据集的tfrecord解析函数，包含了数据增强部分
        :param example_proto:
        :return:
        """
        features = {'image_row': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64)}
        features = tf.io.parse_single_example(example_proto, features)
        img = tf.io.decode_raw(features['image_row'], tf.uint8)
        img = tf.reshape(img, shape=(self.save_shape[0], self.save_shape[1], 3))
        label = tf.cast(features['label'], tf.int64)
        label = tf.one_hot(label, len(self.classes))
        # Augmentation
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.rot90(img, k=random.choice([0,1,2,3]))
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img, label

    def _parse_eval_func(self, example_proto):
        """
        val和test数据集的tfrecord解析函数，不包含数据增强
        :param example_proto:
        :return:
        """
        features = {'image_row': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64)}
        features = tf.io.parse_single_example(example_proto, features)
        img = tf.io.decode_raw(features['image_row'], tf.uint8)
        img = tf.reshape(img, shape=(self.save_shape[0], self.save_shape[1], 3))
        label = tf.cast(features['label'], tf.int64)
        label = tf.one_hot(label, len(self.classes))
        return img, label

if __name__ == '__main__':
    data_dir = '/home/demon/work2/蒙坤测试/difficult'
    tfrecord_dir = 'difficult_tfrecord'
    train_p = 0.8
    val_p = 0.1
    test_p = 0.1
    img_type = ['jpg','jpeg','bmp']
    save_shape = (300,300)

    tfrecord_writer = TFrecordDataGenerate(data_dir,tfrecord_dir,save_shape)
    # tfrecord_writer.write()
    # print(tfrecord_writer.classes)
    data = tfrecord_writer.get_train_data(2)
    for batch_img,batch_label in data:
        for img in batch_img:
            img = img.numpy()
            cv2.imshow('1',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

