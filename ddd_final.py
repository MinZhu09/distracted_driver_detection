# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import glob
import keras
import cv2
from keras.utils import np_utils
from keras.optimizers import *
import random

from math import ceil

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import *

np.random.seed(2017)
data_total_batch = 10
color_type_global = 3
work_path = 'D:\MachineLearing\distracted_driver_detection\distracted_driver_detection'
# work_path = '/Users/minzhu/Documents/MachineLearning/UdacityProjects/FinalProject/distracted_diver_detection'
all_origin_cvs_file = 'driver_imgs_list.csv'
# all_da_cvs_file = 'driver_imgs_list_da.csv'
da_multiple = 31

datagen = ImageDataGenerator(
    featurewise_center=False,  # 使输入数据集去中心化（均值为0）, 按feature执行。
    samplewise_center=False,  # 使输入数据的每个样本均值为0。
    featurewise_std_normalization=False,  # 将输入除以数据集的标准差以完成标准化, 按feature执行
    samplewise_std_normalization=False,  # 将输入的每个样本除以其自身的标准差
    zca_whitening=False,  # 布尔值，对输入数据施加ZCA白化
    rotation_range=15.,  # 整数，数据提升时图片随机转动的角度。随机选择图片的角度，是一个0~180的度数，取值为0~180
    width_shift_range=0.15,  # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
    height_shift_range=0.1,  # 浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度
    shear_range=0.1,
    zoom_range=(0.80, 1.2),
    # 浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]。用来进行随机的放大
    channel_shift_range=10.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,  # 布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
    vertical_flip=False)  # 布尔值，进行随机竖直翻转

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def get_session(gpu_fraction=0.8):
    '''Assume that you have 10GB of GPU memory and want to allocate ~8GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


import matplotlib.pyplot as plt

def draw_training_result(history, modelStr=''):
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    accfig = plt.figure(0, figsize=(12, 5))  # 创建figure窗口
    plt.subplot(1, 2, 1)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # picpath = os.path.join('cache', 'acc_' + modelStr + '.png')
    # plt.savefig(picpath)
    # plt.close(accfig)  # 关闭图 0

    # summarize history for loss
    # lossfig = plt.figure(1)  # 新图 1
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    picpath = os.path.join('cache', 'loss_' + modelStr + '.png')
    plt.savefig(picpath)
    # plt.close(lossfig)  # 关闭图 1
    plt.close(accfig)  # 关闭图 1


def visualize_class_activation_map(Pretrained_model, model_path, final_conv_layer_name, img_path, img_size, target_class,
                                   preprocess_input=None, show_img=False):
    # model = load_model(model_path)
    # 加载模型数据和weights
    # model = model_from_json(open(model_path + '.json').read())
    # model.load_weights(model_path + '.h5')
    base_model = Pretrained_model(weights='imagenet', include_top=False,
                                  input_shape=(image_size[0], image_size[1], color_type_global), classes=10)
    x = top_classifier(input_tensor=base_model.output, add_gap=True, compile_model=False)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(model_path)

    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, img_size)  # (224,224,3)
    # Reshape to the network input shape (w, h, 3).
    x = np.array(original_img, dtype=np.float32)  # (224,224,3)
    x = np.expand_dims(x, axis=0)
    if preprocess_input is not None:
        x = preprocess_input(x)

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]  # (512,10)

    # Get the output, feature maps, of the final conv layer
    final_conv_layer = model.get_layer(name=final_conv_layer_name)  # (?, 14, 14, 512)
    get_output = K.function([model.layers[0].input], [final_conv_layer.output])
    conv_outputs = get_output([x])[0]  # (1,14,14,512)
    conv_outputs = conv_outputs[0, :, :, :]  # (14,14,512)

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])  # (14, 14)
    target_weights = class_weights[:, target_class]  # (512,)
    for i, w in enumerate(target_weights):
        cam += w * conv_outputs[:, :, i]

    y_probs = model.predict(x)  # 预测类的概率 （1, 10）
    y_probs_target = y_probs[:, target_class]  # target类别的预测概率 (1,)
    cam = y_probs_target * cam

    # 调整cam的大小，设定门限：由于只希望将模型认为比较重要的区域标记出来，因此应该选择一个门限值，
    # 将该门限值之下的像素置 0（将该像素透明化）
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam[cam < 0.5] = 0
    # 将cam resize到原图大小
    cam = cv2.resize(cam, img_size)
    cam = np.uint8(255 * cam)

    # 染成彩色。由于ColorMap的工作原理是将任意矩阵的取值范围映射到0~255范围内，因此为了之后好挑选颜色，需要先归一化
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # (224,224,3)

    # 加在原图上
    out = cv2.addWeighted(original_img, 0.8, heatmap, 0.4, 0)  # (224,224,3)

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    if show_img:
        # 显示图片
        # plt.axis('off')
        plt.title('c' + str(target_class) + str(y_probs_target))
        plt.imshow(out, cmap=plt.cm.hsv)
        plt.show()
    return out, y_probs_target


def show_all_classes_cam(Pretrained_model, model_path, final_conv_layer_name, img_size, preprocess_input=None):
    """
    每个label的图片都显示一张
    """
    numRows = 2
    numCols = 5
    plt.figure(figsize=(15, 5))  # 创建figure窗口
    for i in range(0, 10):
        path = os.path.join('imgs', 'train', 'c' + str(i), '*.jpg')
        files = glob.glob(path)
        for file in files:  # 每个类别只显示一个
            cam, prob = visualize_class_activation_map(Pretrained_model=Pretrained_model,
                                                       model_path=model_path,
                                                       final_conv_layer_name=final_conv_layer_name,
                                                       img_path=file,
                                                       img_size=img_size,
                                                       target_class=i,
                                                       preprocess_input=preprocess_input,
                                                       show_img=False)
            # file_basename = os.path.basename(file)

            # img = cv2.cvtColor(images[i], cv2.COLOR_BGR2BGRA)
            plt.subplot(numRows, numCols, i + 1)  # 将窗口分成numRows行，numCols列
            plt.imshow(cam, cmap=plt.cm.hsv)
            plt.title('c' + str(i) + '  ' + str(prob))
            plt.xticks([]), plt.yticks([])
            break

    plt.show()


def top_classifier(input_shape=None, input_tensor=None, add_gap=True, compile_model=True):
    '''
    获取top_model，根据需要选择采用input_shape或input_tensor，
    :param input_shape: 需要返回model时使用该参数
    :param input_tensor: 不需要返回model时使用该参数，因为需要加到base_model顶部，必须匹配
    :param compile_model: True-返回model，False-返回tensor
    :return:
    '''
    # '''
    # img_input = Input(shape=input_shape)
    # x = Flatten(input_shape=input_shape)(img_input)

    # x = Input(shape=input_shape)  # 函数式模型

    if input_tensor is not None:
        img_input = input_tensor
    else:
        img_input = Input(shape=input_shape)

    # x = Flatten()(img_input)
    # x = Dense(1024, activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4),
    #           bias_regularizer=regularizers.l2(1e-4))(x)  # has weights
    # x = Dropout(0.7)(x)
    # x = BatchNormalization()(x)  # has weights
    # x = LeakyReLU(alpha=0.0001)(x)

    # 使用GAP代替FC
    if add_gap:
        x = GlobalAveragePooling2D()(img_input)
    else:
        x = img_input

    x = Dropout(0.7)(x)

    # # InceptionV3过拟合了  添加BN试试
    # x = BatchNormalization()(x)  # has weights
    # x = LeakyReLU(alpha=0.0001)(x)

    x = Dense(10, activation="softmax",
              kernel_initializer=keras.initializers.he_normal(seed=2017), kernel_regularizer=regularizers.l2(1e-4),
              bias_initializer=keras.initializers.he_normal(seed=2018), bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(x)  # has weights

    if compile_model:
        model = Model(inputs=img_input,
                      outputs=x,
                      name='vgg16_fc_top')

        print('top_model.summary()')
        # print(model.summary())

        # Learning rate is changed to 0.001
        # sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=sgd, loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        model.compile(optimizer=Adam(lr=1e-4, decay=1e-6), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
    else:
        return x


def fine_tuned_model(Pretrained_model, img_rows, img_cols, freeze_layers_num=15,
                     top_classifier_weights_path=''):
    # notop: 19 layers (no flatten, dense, dropout); withtop:  layers
    nb_classes = 10
    base_model = Pretrained_model(weights='imagenet', include_top=False,
                                  input_shape=(img_rows, img_cols, color_type_global), classes=nb_classes)

    x = top_classifier(input_tensor=base_model.output, add_gap=True, compile_model=False)
    model = Model(inputs=base_model.input, outputs=x)

    start_layer_no = len(base_model.layers)
    print('load top weights : ', top_classifier_weights_path + '.hdf5')
    f = h5py.File(top_classifier_weights_path + '.hdf5')
    topology.load_weights_from_hdf5_group(f=f, layers=model.layers[start_layer_no:])

    # fine-tuning
    # setting the first 15 layers to non-trainable
    # (the original weights will not be updated)
    print('freeze layers num :', freeze_layers_num)
    for layer in model.layers[:freeze_layers_num]:
        layer.trainable = False

    # Compiling the model with a SGD/momentum optimizer
    # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4, decay=1e-6), metrics=['accuracy'])  # adam作用？

    print(model.summary())

    return model


import gc


def collect_and_show_garbage():
    # print('collecting...')
    n = gc.collect()
    gc.garbage
    # print('unreachle objects:', n)
    # print(gc.garbage)


# windows没有resources模块,使用psutil代替
# import resources
class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # print('\nepoch memory: ', pkg_resources.getrusage(pkg_resources.RUSAGE_SELF).ru_maxrss)
        collect_and_show_garbage()
        # print(u'  内存使用：', psutil.Process(os.getpid()).memory_info().rss)

        # def on_batch_end(self, batch, logs=None):
        #     # print('\nbatch memory: ', pkg_resources.getrusage(pkg_resources.RUSAGE_SELF).ru_maxrss)
        #     print(u'  内存使用：', psutil.Process(os.getpid()).memory_info().rss)


def all_unique_drivers():
    path = os.path.join(work_path, 'driver_imgs_list.csv')
    cvs_lines = pd.read_csv(path)
    unique_drivers = np.array(list((set(cvs_lines['subject']))))
    return unique_drivers


def split_test_valid_on_driver(cvs_file_path, save_to_path_prefix):
    unique_drivers = all_unique_drivers()
    # random split
    # np.random.seed(fold_number)
    split_data = np.random.uniform(0.08, 0.12)
    train_driver_number = int(len(unique_drivers) * (1 - split_data))
    # train_driver_number = 26 - 2
    alltrain_drivers = unique_drivers[:train_driver_number]
    allvalid_drivers = unique_drivers[train_driver_number:]
    print('Split train drivers: ', alltrain_drivers)
    print('Split valid drivers: ', allvalid_drivers)
    # 随机打乱driver ID
    np.random.shuffle(unique_drivers)

    # 获取到 driverId class imageName
    path = os.path.join(work_path, all_origin_cvs_file)
    f = open(path)
    origin_cvs_lines = list(f)
    f.close()
    random.shuffle(origin_cvs_lines)

    valid_images = []
    train_images = []
    for line in origin_cvs_lines:
        line = line.replace('\n', '').split(',')
        img_path = os.path.join(work_path, 'imgs', 'train', line[1], line[2])
        if os.path.exists(img_path) == False:
            continue

        if line[0] in allvalid_drivers:
            valid_images.append(line)
        else:
            train_images.append(line)
    valid_cvs_path = os.path.join(work_path, save_to_path_prefix + 'driver_imgs_list_allvalid.csv')
    valid_images_df = pd.DataFrame(valid_images, columns=("subject", "classname", "img"))
    valid_images_df.to_csv(save_to_path_prefix + 'driver_imgs_list_allvalid.csv', index=False)

    # path = os.path.join(work_path, new2_da_cvs_file)
    # f = open(path)
    # da_cvs_lines = list(f)
    # f.close()
    # random.shuffle(da_cvs_lines)
    #
    # for line in da_cvs_lines:
    #     line = line.replace('\n', '').split(',')
    #     img_path = os.path.join(work_path, 'imgs', 'train', line[1], line[2])
    #     if os.path.exists(img_path) == False:
    #         continue
    #
    #     if line[0] in alltrain_drivers:
    #         train_images.append(line)
    train_cvs_path = os.path.join(work_path, save_to_path_prefix + 'driver_imgs_list_alltrain.csv')
    train_images_df = pd.DataFrame(train_images, columns=("subject", "classname", "img"))
    train_images_df.to_csv(save_to_path_prefix + 'driver_imgs_list_alltrain.csv', index=False)

    return train_cvs_path, len(train_images), valid_cvs_path, len(valid_images)


def cv_split_test_valid(cvs_file_path, save_to_path_prefix, kfold):  # kfold < 9
    unique_drivers = all_unique_drivers()
    # 随机打乱driver ID
    np.random.shuffle(unique_drivers)

    path = os.path.join(work_path, all_origin_cvs_file)
    f = open(path)
    origin_cvs_lines = list(f)
    f.close()
    np.random.shuffle(origin_cvs_lines)

    # path = os.path.join(work_path, new2_da_cvs_file)
    # f = open(path)
    # da_cvs_lines = list(f)
    # f.close()
    # np.random.shuffle(da_cvs_lines)

    train_cvs_path_list = []
    train_img_len_list = []
    valid_cvs_path_list = []
    valid_img_len_list = []
    for i in range(kfold):
        allvalid_drivers = unique_drivers[i * 3:(i + 1) * 3]
        alltrain_drivers = list(set(unique_drivers) - set(allvalid_drivers))
        print('Split train drivers: ', alltrain_drivers)
        print('Split valid drivers: ', allvalid_drivers)

        valid_images = []
        train_images = []
        for line in origin_cvs_lines:
            line = line.replace('\n', '').split(',')
            img_path = os.path.join(work_path, 'imgs', 'train', line[1], line[2])
            if os.path.exists(img_path) == False:
                continue

            if line[0] in allvalid_drivers:
                valid_images.append(line)
            else:
                train_images.append(line)

        # save
        valid_cvs_path = os.path.join(work_path, save_to_path_prefix + 'driver_imgs_list_allvalid' + str(i) + '.csv')
        valid_images_df = pd.DataFrame(valid_images, columns=("subject", "classname", "img"))
        valid_images_df.to_csv(valid_cvs_path, index=False)
        valid_cvs_path_list.append(valid_cvs_path)
        valid_img_len_list.append(len(valid_images))


        # for line in da_cvs_lines:
        #     line = line.replace('\n', '').split(',')
        #     img_path = os.path.join(work_path, 'imgs', 'train', line[1], line[2])
        #     if os.path.exists(img_path) == False:
        #         continue
        #
        #     if line[0] in alltrain_drivers:
        #         train_images.append(line)

        # save
        train_cvs_path = os.path.join(work_path, save_to_path_prefix + 'driver_imgs_list_alltrain' + str(i) + '.csv')
        train_images_df = pd.DataFrame(train_images, columns=("subject", "classname", "img"))
        train_images_df.to_csv(train_cvs_path, index=False)
        train_cvs_path_list.append(train_cvs_path)
        train_img_len_list.append(len(train_images))


    return train_cvs_path_list, train_img_len_list, valid_cvs_path_list, valid_img_len_list


def process_data_and_target(X_train, input_shape, y_train):
    X_train = np.array(X_train, dtype=np.float32)
    X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1],
                              input_shape[2])

    y_train = np.array(y_train, dtype=np.uint8)
    y_train = np_utils.to_categorical(y_train, 10)

    return X_train, y_train


def image_augmentation(X_train, Y_train, batch_size, datagen=datagen):
    # fits the model on batches with real-time data augmentation:
    datagen.fit(X_train)
    return datagen.flow(X_train, Y_train, batch_size=batch_size).next()


def da_with_random_image(origin_file, label, img_cols, img_rows):
    origin_image = cv2.imread(origin_file)
    center_path = os.path.join(work_path, 'imgs', 'train', 'c' + str(label), '*.jpg')
    center_files = glob.glob(center_path)

    random_samples = random.sample(center_files, da_multiple * 2)
    results = []
    for i in range(da_multiple):
        left_file = random_samples[i]
        left_img = cv2.imread(left_file)

        right_file = random_samples[i+da_multiple]
        right_img = cv2.imread(right_file)

        # 左-中-右结合
        left_tmp = left_img[0:480, 0:90]
        center_tmp = origin_image[0:480, 90:480]
        right_tmp = right_img[0:480, 480:640]
        img_mix = np.concatenate((left_tmp, center_tmp, right_tmp), axis=1)

        # img_mix = do_gamma_trans(img_mix)
        img_mix = cv2.resize(img_mix, (img_cols, img_rows))

        img_mix = np.array(img_mix, dtype=np.float32)
        img_mix = np.expand_dims(img_mix, axis=0)

        # print('da ' + str(label) + '  : ' + left_file + '  ' + origin_file + '  ' + right_file)
        results.append(img_mix)

        # cv2.imwrite(filename=os.path.join(work_path, 'test_da2.jpg'),
        #             img=img_mix)

    return results

def data_generator_from_file(path, batch_size, img_rows=224, img_cols=224,
                             isvalidation=False, preprocess=None):
    while 1:
        f = open(path)
        cvs_lines = list(f)
        f.close()
        # if isvalidation == False:
        #     random.shuffle(cvs_lines)

        batch_index = 0  # 记录当前图片在当前batch中的index，到batch_size后会重新计算
        current_batch = 0
        for line in cvs_lines:  # 对Training set里的每张图片做数据增强，batch_size张图片后返回x,y
            if batch_index == 0:
                X_train = []
                y_train = []

            line = line.replace('\n', '').split(',')
            if len(line) < 3:
                continue
            # create numpy arrays of input data
            # and labels, from each line in the file
            img_path = os.path.join(work_path, 'imgs', 'train', line[1], line[2])
            if os.path.exists(img_path) == False:
                continue

            # x, y = get_im_from_cvs_line(line, color_type=color_type)
            # if isvalidation == False:
            #     # gamma trans
            #     if gamma_trans == True:
            #         x = do_gamma_trans(x)

            y = int(line[1][1:])

            img = load_img(img_path, target_size=(img_rows, img_cols))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            if preprocess is not None:
                x = preprocess(x)

            X_train.append(x)
            y_train.append(y)
            batch_index += 1

            # # 可以做一些数据增强
            # if isvalidation == False:
            #     da_results = da_with_random_image(img_path, y, img_cols, img_rows)
            #     for da_x in da_results:
            #         # da_x = np.array(da_x, dtype=np.float32)
            #         if preprocess is not None:
            #             da_x = preprocess(da_x)
            #
            #         X_train.append(da_x)
            #         y_train.append(y)
            #         batch_index += 1

            if batch_index % batch_size == 0:
                X_train, y_train = process_data_and_target(X_train, (img_rows, img_cols, 3), y_train)

                if isvalidation == False and current_batch%2 == 0:
                    X_train, y_train = image_augmentation(X_train, y_train, batch_index)

                # init
                batch_index = 0
                current_batch += 1
                yield (X_train, y_train)

        else:  # 剩下的图片数不够batch_size，做完处理返回x,y

            X_train, y_train = process_data_and_target(X_train, (img_rows, img_cols, 3), y_train)

            if isvalidation == False:
                X_train, y_train = image_augmentation(X_train, y_train, batch_index)

            yield (X_train, y_train)

def get_y_from_cvs_file(path):
    path = os.path.join(work_path, path)
    f = open(path)
    cvs_lines = list(f)
    f.close()

    yfull = []
    for line in cvs_lines:
        line = line.replace('\n', '').split(',')
        img_path = os.path.join(work_path, 'imgs', 'train', line[1], line[2])
        if len(line) < 3 or os.path.exists(img_path)==False:
            continue
        # create numpy arrays of input data
        # and labels, from each line in the file
        y = int(line[1][1:])
        # y = np_utils.to_categorical(y, 10)

        yfull.append(y)

    yfull = np.array(yfull, dtype=np.uint8)
    yfull = np_utils.to_categorical(yfull, 10)

    return yfull


def get_test_numbers():
    path = os.path.join('imgs', 'test', '*.jpg')
    files = glob.glob(path)
    batch_num = 0
    for file in files:
        batch_num += 1

    return batch_num


def test_data_generator_from_path(batch_size,
                                  img_rows=224, img_cols=224,
                                  preprocess=None):
    while 1:
        batch_index = 0  # 记录当前图片在当前batch中的index，到batch_size后会重新计算
        path = os.path.join('imgs', 'test', '*.jpg')
        files = glob.glob(path)

        current_batch = 0
        for file in files:
            if batch_index == 0:
                X_train = []

            # x = get_im_cv2(file, img_rows, img_cols, color_type)
            img = load_img(file, target_size=(img_rows, img_cols))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            if preprocess is not None:
                x = preprocess(x)

            X_train.append(x)
            batch_index += 1

            if batch_index % batch_size == 0:
                # current_batch += 1
                # print("predict batch: ", current_batch)
                # X_train = normalize_batch_data_without_label(X_train)
                X_train = np.array(X_train, dtype=np.float32)
                X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)

                # init
                batch_index = 0
                yield X_train

        else:  # 剩下的图片数不够batch_size，做完处理返回x,y
            # print("predict the last batch")
            # X_train = normalize_batch_data_without_label(X_train)
            X_train = np.array(X_train, dtype=np.float32)
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)

            yield X_train

def save_pred(preds, info):
    print('Read test images name for submission file')
    path = os.path.join('imgs', 'test', '*.jpg')
    files = glob.glob(path)
    X_test_id = []

    for file in files:
        X_test_id.append(os.path.basename(file))

    ## error: 'list' object has no attribute 'clip'
    preds = preds.clip(min=0.05, max=0.995)  # logLoss处理无穷大的问题
    preds_df = pd.DataFrame(preds, columns=['c0', 'c1', 'c2', 'c3',
                                            'c4', 'c5', 'c6', 'c7',
                                            'c8', 'c9'])
    preds_df['img'] = X_test_id

    print('Saving predictions')
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    preds_df.to_csv(sub_file, index=False)


# def predict_data_on_trained_model(image_size, batch_size=128, nb_epoch=8, modelStr='', save_predicts=True,
#                                   preprocess=None):
#     # img_rows, img_cols = 224, 224
#     img_rows = image_size[0]
#     img_cols = image_size[1]
#     # batch_size = 128
#
#     # 加载模型数据和weights
#     model = model_from_json(open(modelStr + '.json').read())
#     model.load_weights(modelStr + '.h5')
#
#     # predict
#     test_samples = get_test_numbers()
#     print('test_steps_per_epoch: ', ceil(test_samples / batch_size))
#     test_data_generator = test_data_generator_from_path(batch_size, img_rows, img_cols, preprocess=preprocess)
#     y_test = model.predict_generator(generator=test_data_generator,
#                                      steps=ceil(test_samples / batch_size),
#                                      verbose=1)
#
#     # 释放资源
#     keras.backend.clear_session()
#     collect_and_show_garbage()
#
#     # 保存预测结果
#     if save_predicts:
#         info_string = modelStr + '_kfold_' + str(1) \
#                       + '_ep_' + str(nb_epoch)
#         save_pred(y_test, info_string)
#     else:
#         return y_test


def bottleneck_model(MODEL, image_size, batch_size=128, nb_epoch=8, modelStr='',
                     preprocess=None):
    # Get bottleneck features of VGG16 model
    img_rows = image_size[0]
    img_cols = image_size[1]

    ### 尝试按driverID来划分  分割train、valid，并保存起来，在train top-model时需要使用
    train_cvs_path, train_number, valid_cvs_path, valid_number \
        = split_test_valid_on_driver(all_origin_cvs_file, modelStr)

    print('Split train: ', train_number)
    print('Split valid: ', valid_number)

    # Get bottleneck features
    starttime = datetime.datetime.now()
    base_model = MODEL(include_top=False,
                       weights='imagenet',
                       input_shape=(img_rows, img_cols, color_type_global))
    # 利用 GlobalAveragePooling2D 将卷积层输出的每个激活图直接求平均值，不然输出的文件会非常大，且容易过拟合
    # 之前没加,导致 InceptionV3 使用DA后的数据集训练时总是 out of memory
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    train_data_generator = data_generator_from_file(path=train_cvs_path, batch_size=batch_size,
                                                    img_rows=img_rows, img_cols=img_cols,
                                                    isvalidation=True,
                                                    preprocess=preprocess)
    bottleneck_train_y = model.predict_generator(generator=train_data_generator,
                                                 steps=ceil(train_number / batch_size),
                                                 verbose=1)
    # Get valid bottleneck features
    valid_data_generator = data_generator_from_file(path=valid_cvs_path, batch_size=batch_size,
                                                    img_rows=img_rows, img_cols=img_cols,
                                                    isvalidation=True,
                                                    preprocess=preprocess)
    bottleneck_valid_y = model.predict_generator(generator=valid_data_generator,
                                                 steps=ceil(valid_number / batch_size),
                                                 verbose=1)

    keras.backend.clear_session()
    collect_and_show_garbage()

    endtime = datetime.datetime.now()
    print('get bottleneck-features time: ', (endtime - starttime).seconds)
    starttime = endtime

    print('bottleneck train: ', len(bottleneck_train_y))
    print('bottleneck valid: ', len(bottleneck_valid_y))

    # train top-model
    top_model = top_classifier(input_shape=model.output_shape[1:], add_gap=False, compile_model=True)

    bf_y_train = get_y_from_cvs_file(train_cvs_path)
    bf_y_valid = get_y_from_cvs_file(valid_cvs_path)

    earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4)
    reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1,
                                          verbose=1, min_lr=1e-7)
    modelCheckpoint = ModelCheckpoint(filepath=modelStr + '.hdf5', verbose=1,
                                      save_best_only=True, save_weights_only=True, mode='auto')
    hist = top_model.fit(x=bottleneck_train_y, y=bf_y_train, batch_size=batch_size, epochs=nb_epoch,
                         verbose=2,
                         callbacks=[MemoryCallback(), earlystop, reduceLROnPlateau, modelCheckpoint],
                         validation_data=(bottleneck_valid_y, bf_y_valid))

    # top_model.save_weights(filepath=modelStr + '.hdf5')

    keras.backend.clear_session()
    collect_and_show_garbage()

    # 绘制训练过程
    draw_training_result(history=hist, modelStr=modelStr)


# def run_without_cross_validation(MODEL, freeze_layers_num, image_size, batch_size=128, nb_epoch=8, modelStr='',
#                                  top_classifier_weights_path='', preprocess=None):
#     img_rows = image_size[0]
#     img_cols = image_size[1]
#
#     # 创建文件夹，保存模型
#     if not os.path.isdir('cache'):
#         os.mkdir('cache')
#
#     ## 尝试按driverID来划分
#     train_cvs_path, train_number, valid_cvs_path, valid_number \
#         = split_test_valid_on_driver(all_origin_cvs_file, modelStr)
#
#     print('Split train: ', train_number)
#     print('Split valid: ', valid_number)
#
#     train_data_generator = data_generator_from_file(train_cvs_path, batch_size=batch_size,
#                                                     img_rows=img_rows, img_cols=img_cols,
#                                                     isvalidation=False,
#                                                     preprocess=preprocess)
#     valid_data_generator = data_generator_from_file(valid_cvs_path, batch_size=batch_size,
#                                                     img_rows=img_rows, img_cols=img_cols,
#                                                     isvalidation=True,
#                                                     preprocess=preprocess)
#
#     model = fine_tuned_model(Pretrained_model=MODEL, freeze_layers_num=freeze_layers_num,
#                              img_rows=img_rows, img_cols=img_cols,
#                              top_classifier_weights_path=top_classifier_weights_path)
#     log_path = os.path.join('logs', 'nocv')
#     tensorboard = TensorBoard(log_dir=log_path)
#     earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=1e-5)
#     reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1,
#                                           verbose=1, min_lr=1e-7)  # 减少learning rate
#     modelCheckpoint = ModelCheckpoint(filepath=modelStr + '.hdf5', verbose=1,
#                                       save_best_only=True, save_weights_only=True, mode='auto')
#     # trianing
#     print('steps_per_epoch: ', ceil(train_number / batch_size))
#     print('validation_steps: ', ceil(valid_number / batch_size))
#     hist = model.fit_generator(
#         generator=train_data_generator,
#         steps_per_epoch=ceil(train_number / batch_size),
#         epochs=nb_epoch,
#         verbose=2,
#         # callbacks=[MemoryCallback(), tensorboard, earlystop, reduceLROnPlateau, modelCheckpoint],
#         callbacks=[MemoryCallback(), tensorboard, earlystop],
#         # , changelearningrate],
#         validation_data=valid_data_generator,
#         validation_steps=ceil(valid_number / batch_size),
#         shuffle=True)
#
#     # 绘制训练过程
#     draw_training_result(history=hist, modelStr=modelStr)
#
#     # model.save_weights(filepath=modelStr + '.hdf5')
#     json_string = model.to_json()  # 等价于 json_string = model.get_config()
#     open(modelStr + '.json', 'w').write(json_string)
#     model.save_weights(modelStr + '.h5')
#
#     # predict
#     # model.load_weights(filepath=modelStr + '.hdf5')
#     test_samples = get_test_numbers()
#     print('test_steps_per_epoch: ', ceil(test_samples / batch_size))
#     test_data_generator = test_data_generator_from_path(batch_size, img_rows, img_cols, preprocess=preprocess)
#     y_test = model.predict_generator(generator=test_data_generator,
#                                      steps=ceil(test_samples / batch_size),
#                                      verbose=1)
#
#     # 释放资源
#     keras.backend.clear_session()
#     collect_and_show_garbage()
#
#     # 保存预测结果
#     info_string = modelStr + '_kfold_' + str(1) + '_ep_' + str(nb_epoch)
#     save_pred(y_test, info_string)


def run_cross_validation(MODEL, freeze_layers_num, image_size, batch_size=128, kfold=5, nb_epoch=8, modelStr='',
                                 top_classifier_weights_path='', preprocess=None):
    img_rows = image_size[0]
    img_cols = image_size[1]

    # 创建文件夹，保存模型
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    train_cvs_path_list, train_img_len_list, valid_cvs_path_list, valid_img_len_list \
        = cv_split_test_valid(all_origin_cvs_file, modelStr, kfold)

    test_samples = get_test_numbers()
    yfull_predict = np.zeros(shape=(test_samples, 10), dtype=np.float)

    for i in range(kfold):
        train_number = train_img_len_list[i]
        valid_number = valid_img_len_list[i]
        print('Split train: ', train_number)
        print('Split valid: ', valid_number)

        train_cvs_path = train_cvs_path_list[i]
        valid_cvs_path = valid_cvs_path_list[i]

        train_data_generator = data_generator_from_file(train_cvs_path, batch_size=batch_size,
                                                        img_rows=img_rows, img_cols=img_cols,
                                                        isvalidation=False,
                                                        preprocess=preprocess)
        valid_data_generator = data_generator_from_file(valid_cvs_path, batch_size=batch_size,
                                                        img_rows=img_rows, img_cols=img_cols,
                                                        isvalidation=True,
                                                        preprocess=preprocess)

        model = fine_tuned_model(Pretrained_model=MODEL, freeze_layers_num=freeze_layers_num,
                                 img_rows=img_rows, img_cols=img_cols,
                                 top_classifier_weights_path=top_classifier_weights_path)
        log_path = os.path.join('logs', 'cv')
        tensorboard = TensorBoard(log_dir=log_path)
        earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4)
        reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, cooldown=1,
                                              verbose=1, min_lr=1e-7)  # 减少learning rate
        modelCheckpoint = ModelCheckpoint(filepath=modelStr + '.fold_' + str(i) + '.hdf5', verbose=1,
                                          save_best_only=True, save_weights_only=True, mode='auto')
        # trianing
        print('steps_per_epoch: ', ceil(train_number * (da_multiple + 1) / batch_size))
        print('validation_steps: ', ceil(valid_number / batch_size))
        hist = model.fit_generator(
            generator=train_data_generator,
            steps_per_epoch=ceil(train_number * (da_multiple + 1) / batch_size),
            epochs=nb_epoch,
            verbose=1,
            callbacks=[MemoryCallback(), tensorboard, earlystop, reduceLROnPlateau, modelCheckpoint],
            # changelearningrate],
            validation_data=valid_data_generator,
            validation_steps=ceil(valid_number / batch_size),
            shuffle=True)

        # 绘制训练过程
        draw_training_result(history=hist, modelStr=modelStr + '.fold_' + str(i))

        # predict
        model.load_weights(filepath=modelStr + '.fold_' + str(i) + '.hdf5')
        # model.save_weights(filepath=modelStr + '.fold_' + str(i) + '.hdf5')

        print('test_steps_per_epoch: ', ceil(test_samples / batch_size))
        test_data_generator = test_data_generator_from_path(batch_size, img_rows, img_cols, preprocess=preprocess)
        y_test = model.predict_generator(generator=test_data_generator,
                                         steps=ceil(test_samples / batch_size),
                                         verbose=1)

        # 释放资源
        keras.backend.clear_session()
        collect_and_show_garbage()

        yfull_predict += y_test


    # 保存预测结果
    info_string = modelStr + '_kfold_' + str(kfold) + '_ep_' + str(nb_epoch)
    yfull_predict /= kfold
    save_pred(yfull_predict, info_string)


def train_on_single_model(Pretrained_MODEL, freeze_layers_num, image_size, batch_size=128, kfold=1,
                          bf_epoch=50, ft_epoch=35, modelStr='', preprocess=None):
    bottleneck_model(MODEL=Pretrained_MODEL,
                     image_size=image_size, batch_size=batch_size,
                     nb_epoch=bf_epoch,
                     modelStr=Pretrained_MODEL.__name__ + modelStr + 'bf',
                     preprocess=preprocess)

    # run_without_cross_validation(MODEL=Pretrained_MODEL, freeze_layers_num=freeze_layers_num,
    #                              image_size=image_size, batch_size=batch_size,
    #                              nb_epoch=ft_epoch,
    #                              modelStr=Pretrained_MODEL.__name__ + modelStr + 'ft',
    #                              top_classifier_weights_path=Pretrained_MODEL.__name__ + modelStr + 'bf',
    #                              preprocess=preprocess)

    run_cross_validation(MODEL=Pretrained_MODEL, freeze_layers_num=freeze_layers_num,
                         image_size=image_size, batch_size=batch_size,
                         kfold=kfold,
                         nb_epoch=ft_epoch,
                         modelStr=Pretrained_MODEL.__name__ + modelStr + 'ft',
                         top_classifier_weights_path=Pretrained_MODEL.__name__ + modelStr + 'bf',
                         preprocess=preprocess)

def predict_data_on_trained_model(MODEL, image_size, batch_size=128, nb_epoch=1, modelStr='', save_predicts=True, preprocess=None):
    img_rows = image_size[0]
    img_cols = image_size[1]

    # 加载模型数据和weights
    # model = model_from_json(open(modelStr + '.json').read())
    base_model = MODEL(weights='imagenet', include_top=False,
                       input_shape=(img_rows, img_cols, color_type_global), classes=10)
    x = top_classifier(input_tensor=base_model.output, add_gap=True, compile_model=False)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(filepath=modelStr + '.hdf5')

    # predict
    test_samples = get_test_numbers()
    print('test_steps_per_epoch: ', ceil(test_samples / batch_size))
    test_data_generator = test_data_generator_from_path(batch_size, img_rows, img_cols,
                                                        preprocess=preprocess)
    y_test = model.predict_generator(generator=test_data_generator,
                                     steps=ceil(test_samples / batch_size),
                                     verbose=1)

    # 释放资源
    keras.backend.clear_session()
    collect_and_show_garbage()

    # 保存预测结果
    if save_predicts:
        info_string = modelStr + '_kfold_' + str(1) \
                      + '_ep_' + str(nb_epoch)
        save_pred(y_test, info_string)
    else:
        return y_test


def ensemble_models(pretraind_model=[], model_path=[], image_size=[], preprocess=[]):
    test_samples = get_test_numbers()
    yfull_predict = np.zeros(shape=(test_samples, 10), dtype=np.float)
    model_num = len(model_path)
    for i in range(model_num):
        # model_path = model_path[i]
        # image_size = image_size[i]
        # preprocess = preprocess[i]
        # base_model = pretraind_model[i]

        y_test = predict_data_on_trained_model(MODEL=pretraind_model[i], image_size=image_size[i], modelStr=model_path[i],
                                               save_predicts=False, preprocess=preprocess[i])
        yfull_predict += y_test

    # 保存预测结果
    info_string = 'ensemble_results'
    yfull_predict /= len(model_path)
    save_pred(yfull_predict, info_string)

def write_gap(MODEL, image_size, train_cvs_path, train_number, valid_cvs_path, valid_number, preprocess=None):
    # Get bottleneck features of VGG16 model
    img_rows = image_size[0]
    img_cols = image_size[1]
    batch_size = 128

    # ### 尝试按driverID来划分  分割train、valid，并保存起来，在train top-model时需要使用
    # train_cvs_path, train_number, valid_cvs_path, valid_number \
    #     = split_test_valid_on_driver(all_origin_cvs_file, MODEL.__name__)
    # bf_y_train = get_y_from_cvs_file(train_cvs_path)
    # bf_y_valid = get_y_from_cvs_file(valid_cvs_path)
    #
    # print('Split train: ', train_number)
    # print('Split valid: ', valid_number)

    # Get bottleneck features
    base_model = MODEL(include_top=False,
                       weights='imagenet',
                       input_shape=(img_rows, img_cols, color_type_global))
    # 利用 GlobalAveragePooling2D 将卷积层输出的每个激活图直接求平均值，不然输出的文件会非常大，且容易过拟合
    # 之前没加,导致 InceptionV3 使用DA后的数据集训练时总是 out of memory
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    train_data_generator = data_generator_from_file(path=train_cvs_path, batch_size=batch_size,
                                                    img_rows=img_rows, img_cols=img_cols,
                                                    isvalidation=False,
                                                    preprocess=preprocess)
    bottleneck_train_y = model.predict_generator(generator=train_data_generator,
                                                 steps=ceil(train_number / batch_size),
                                                 verbose=1)
    # Get valid bottleneck features
    valid_data_generator = data_generator_from_file(path=valid_cvs_path, batch_size=batch_size,
                                                    img_rows=img_rows, img_cols=img_cols,
                                                    isvalidation=False,
                                                    preprocess=preprocess)
    bottleneck_valid_y = model.predict_generator(generator=valid_data_generator,
                                                 steps=ceil(valid_number / batch_size),
                                                 verbose=1)

    test_samples = get_test_numbers()
    print('test_steps_per_epoch: ', ceil(test_samples / batch_size))
    test_data_generator = test_data_generator_from_path(batch_size, img_rows, img_cols, preprocess=preprocess)
    y_test = model.predict_generator(generator=test_data_generator,
                                     steps=ceil(test_samples / batch_size),
                                     verbose=1)

    with h5py.File("gap_%s.h5"%MODEL.__name__) as h:
        h.create_dataset("x_train", data=bottleneck_train_y)
        # h.create_dataset("y_train", data=bf_y_train)
        h.create_dataset("x_valid", data=bottleneck_valid_y)
        # h.create_dataset("y_valid", data=bf_y_valid)
        h.create_dataset("x_test", data=y_test)

    keras.backend.clear_session()
    collect_and_show_garbage()

def ensemble_from_bf():
    ### 尝试按driverID来划分  分割train、valid，并保存起来，在train top-model时需要使用
    train_cvs_path, train_number, valid_cvs_path, valid_number \
        = split_test_valid_on_driver(all_origin_cvs_file, 'ensemble_bf_')
    # '''
    # Split train drivers:  ['p035' 'p014' 'p061' 'p002' 'p050' 'p056' 'p041' 'p022' 'p012' 'p075'
    # 'p042' 'p049' 'p021' 'p066' 'p039' 'p064' 'p024' 'p015' 'p016' 'p072'
    # 'p081' 'p045' 'p052']
    # Split valid drivers:  ['p026' 'p047' 'p051']
    # Split train:  19472
    # Split valid:  2951
    # '''
    # train_cvs_path = 'ensemble_bf_driver_imgs_list_alltrain.csv'#get_y_from_cvs_file(train_cvs_path)
    # valid_cvs_path = 'ensemble_bf_driver_imgs_list_allvalid.csv'#get_y_from_cvs_file(valid_cvs_path)
    # train_number = 19472
    # valid_number = 2951

    print('Split train: ', train_number)
    print('Split valid: ', valid_number)

    write_gap(ResNet50, (224, 224), train_cvs_path, train_number, valid_cvs_path, valid_number, resnet50.preprocess_input)
    write_gap(InceptionV3, (299, 299), train_cvs_path, train_number, valid_cvs_path, valid_number, inception_v3.preprocess_input)
    write_gap(Xception, (299, 299), train_cvs_path, train_number, valid_cvs_path, valid_number, xception.preprocess_input)

    x_train = []
    # y_train = []
    x_valid = []
    # y_valid = []
    x_test = []

    for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
    # for filename in ["gap_ResNet50.h5"]:
        with h5py.File(filename, 'r') as h:
            x_train.append(np.array(h['x_train']))
            # y_train.append(np.array(h['y_train']))
            x_valid.append(np.array(h['x_valid']))
            # y_valid.append(np.array(h['y_valid']))
            x_test.append(np.array(h['x_test']))

    x_train = np.concatenate(x_train, axis=1)
    # y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=1)
    # y_valid = np.concatenate(y_valid, axis=0)
    x_test = np.concatenate(x_test, axis=1)

    y_train = get_y_from_cvs_file(train_cvs_path)
    y_valid = get_y_from_cvs_file(valid_cvs_path)

    input_tensor = Input(x_train.shape[1:])
    x = Dropout(0.7)(input_tensor)
    x = Dense(10, activation="softmax",
              kernel_initializer=keras.initializers.he_normal(seed=2017), kernel_regularizer=regularizers.l2(1e-4),
              bias_initializer=keras.initializers.he_normal(seed=2017), bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(x)
    model = Model(input_tensor, x)

    model.compile(optimizer=Adam(lr=1e-4, decay=1e-6), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # log_path = os.path.join('logs', 'nocv')
    # tensorboard = TensorBoard(log_dir=log_path)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=1e-5)
    reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1,
                                          verbose=1, min_lr=1e-7)  # 减少learning rate
    modelCheckpoint = ModelCheckpoint(filepath='ensemble_bf' + '.hdf5', verbose=1,
                                      save_best_only=True, save_weights_only=True, mode='auto')
    hist = model.fit(x=x_train, y=y_train, batch_size=128, epochs=100,
                     verbose=2,
                     # callbacks=[MemoryCallback(), earlystop, reduceLROnPlateau, modelCheckpoint],
                     validation_data=(x_valid, y_valid))
    model.save_weights(filepath='ensemble_bf' + '.hdf5')

    # 绘制训练过程
    draw_training_result(history=hist, modelStr='ensemble_bf')

    # model.load_weights(filepath='ensemble_bf' + '.hdf5')
    y_test = model.predict(x=x_test, batch_size=128, verbose=1)

    # 释放资源
    keras.backend.clear_session()
    collect_and_show_garbage()

    # 保存预测结果
    info_string = 'ensemble_bf'
    save_pred(y_test, info_string)

if __name__ == '__main__':
    KTF.set_session(get_session())

    train_on_single_model(Pretrained_MODEL=VGG16, freeze_layers_num=15,#11
                          image_size=(224, 224),
                          batch_size=128, kfold=8, bf_epoch=100, ft_epoch=35,
                          modelStr='_gap_ver4_',
                          preprocess=vgg16.preprocess_input)

    show_all_classes_cam(Pretrained_model=VGG16,
                         model_path='VGG16_gap_ver4_ft.fold_0.hdf5',
                         final_conv_layer_name='block5_conv3',
                         img_size=(224, 224),
                         preprocess_input=vgg16.preprocess_input)

    # we choose to train the top 3 residual blocks
    train_on_single_model(Pretrained_MODEL=ResNet50, freeze_layers_num=140,#163,#166,
                          image_size=(224, 224),
                          batch_size=128, bf_epoch=100, ft_epoch=35,
                          modelStr='_gap_ver4_',
                          # preprocess=resnet50.preprocess_input
                          )

    # we choose to train the top 5 inception blocks   #165 197 229 249,#280 # 311层
    train_on_single_model(Pretrained_MODEL=InceptionV3, freeze_layers_num=165,
                          image_size=(299, 299),
                          batch_size=128, bf_epoch=100, ft_epoch=35,
                          modelStr='_gap_ver4_',
                          preprocess=inception_v3.preprocess_input)

    # train_on_single_model(Pretrained_MODEL=Xception, freeze_layers_num=106,
    #                       image_size=(299, 299),
    #                       batch_size=128, bf_epoch=100, ft_epoch=35,
    #                       modelStr='_gap_ver4_',
    #                       preprocess=xception.preprocess_input)

    # train_on_single_model(Pretrained_MODEL=InceptionResNetV2, freeze_layers_num=249,
    #                       image_size=(299, 299),
    #                       batch_size=128, bf_epoch=100, ft_epoch=35,
    #                       modelStr='_gap_ver4_',
    #                       preprocess=inception_resnet_v2.preprocess_input)

    # 模型融合1
    pretraind_model = [VGG16, VGG16, VGG16, VGG16, VGG16, VGG16, VGG16, VGG16,
                       ResNet50, InceptionV3]
    model_path = ['VGG16_gap_ver4_ft.fold_0', 'VGG16_gap_ver4_ft.fold_1',
                  'VGG16_gap_ver4_ft.fold_2', 'VGG16_gap_ver4_ft.fold_3',
                  'VGG16_gap_ver4_ft.fold_4', 'VGG16_gap_ver4_ft.fold_5',
                  'VGG16_gap_ver4_ft.fold_6', 'VGG16_gap_ver4_ft.fold_7',
                  'ResNet50_gap_ver4_ft.fold_0',
                  'InceptionV3_gap_ver4_ft.fold_0']
    image_size = [(244,244), (244, 244),
                  (244, 244), (244, 244),
                  (244, 244), (244, 244),
                  (244, 244), (244, 244),
                  (244, 244),
                  (299,299)]
    preprocess_list = [vgg16.preprocess_input, vgg16.preprocess_input,
                       vgg16.preprocess_input, vgg16.preprocess_input,
                       vgg16.preprocess_input, vgg16.preprocess_input,
                       vgg16.preprocess_input, vgg16.preprocess_input,
                       resnet50.preprocess_input,
                       inception_v3.preprocess_input]
    ensemble_models(pretraind_model=pretraind_model, model_path=model_path, image_size=image_size, preprocess=preprocess_list)

    # 模型融合2
    ensemble_from_bf()