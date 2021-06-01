# Author  : WangXiao
# File    : train.py
# Function: train model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from models import densenet,efficientnet,mobileNet,resnet18,resnet18_cbam,xception
from tfrecord_data_generate import TFrecordDataGenerate
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau,EarlyStopping
from utils.utils import ModelCheckPoint,WarmUpCosineDecayScheduler
from train_args import Train_Args



def train(args):
    # -------------实例化tfrecord数据生成器---------
    tfrecord_generater = TFrecordDataGenerate(args.dataset,
                                              args.tfrecord_dir,
                                              args.input_shape[:2],
                                              args.is_tfrecord_generate,
                                              args.train_p,
                                              args.val_p,
                                              args.image_type)
    # ------------模型选择------------------
    # 获取类别数量
    classes = tfrecord_generater.classes
    class_num = len(classes)

    # 选择模型
    if args.model_name == 'densenet':
        model = densenet.Densenet(args.input_shape, classes=class_num)
    elif args.model_name == 'xception':
        model = xception.Xception(args.input_shape, classes=class_num)
    elif args.model_name == 'efficient':
        model = efficientnet.EfficientNetB0(input_shape=args.input_shape, classes=class_num)
    elif args.model_name == 'resnet18':
        model = resnet18.ResNet18(input_shape=args.input_shape, classes=class_num)
    elif args.model_name == 'mobileNet':
        model = mobileNet.MobileNet0(input_shape=args.input_shape, classes=class_num)
    elif args.model_name == 'resnet18_cbam':
        model = resnet18_cbam.ResNet18_CBAM(input_shape=args.input_shape, classes=class_num)

    # 加载模型权重
    if args.is_load_weight:
        model.load_weights(args.weight_path)
    # 如果是fineturn，冻结conv层
    if args.finetune:
        for layer in model.layers:
            layer.trainable = False
            # print(1)
            print(layer.name)
            if layer.name in ['avg_pool', 'max_pool']:
                break

    # ----------callback设置---------------
    model_save_path = os.path.join(args.log_dir, args.model_name,'epoch-{epoch:03d}-loss-{loss:.5f}-acc-{accuracy:.3f}-'    # 模型保存地址
                                                       'val_loss-{val_loss:.5f}-val_acc-{val_accuracy:.3f}.h5')
    checkpoint = ModelCheckPoint(model_save_path, monitor='val_loss',       # checkpoint设置
                                 save_weights_only=False, save_best_only=True,period=1)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir,args.model_name)):
        os.mkdir(os.path.join(args.log_dir,args.model_name))
    logging = TensorBoard(args.log_dir) # tensorboard设置
    early_stopping = EarlyStopping(min_delta=0, patience=50, verbose=1)     # earlystoping设置
    #-----------lr策略设置--------------
    if args.is_cosine_scheduler:
        num_train = tfrecord_generater.total_num * args.train_p
        # 预热期
        warmup_epoch = int(args.epoch * 0.2)
        # 总共的步长
        total_steps = int(args.epoch * num_train / args.batch_size)
        # 预热步长
        warmup_steps = int(warmup_epoch * num_train / args.batch_size)
        # 学习率
        reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=args.learning_rate_base,
                                               total_steps=total_steps,
                                               warmup_learning_rate=1e-4,
                                               warmup_steps=warmup_steps,
                                               hold_base_rate_steps=num_train,
                                               min_learn_rate=1e-6
                                               )

    else:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    callbacks = [logging, checkpoint, reduce_lr, early_stopping]
    # ---------------模型编译--------------
    model.compile(optimizer=args.optimizer, loss=args.loss, metrics=[args.metrics])

    # --------------模型训练---------------
    train_gen = tfrecord_generater.get_train_data(args.batch_size)
    val_gen = tfrecord_generater.get_val_data(args.batch_size)
    model.fit(train_gen,epochs=args.epoch,validation_data=val_gen,callbacks=callbacks)


if __name__ == '__main__':

    args = Train_Args()
    train(args)









