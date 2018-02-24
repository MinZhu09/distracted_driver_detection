# -*- coding: utf-8 -*-

import pandas as pd
import glob
import cv2
from keras.utils import np_utils
import random

from keras.preprocessing.image import *
from keras.callbacks import *

work_path = 'D:\MachineLearing\distracted_driver_detection\distracted_driver_detection'
all_origin_cvs_file = 'driver_imgs_list.csv'
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

def all_unique_drivers():
    path = os.path.join(work_path, 'driver_imgs_list.csv')
    cvs_lines = pd.read_csv(path)
    unique_drivers = np.array(list((set(cvs_lines['subject']))))
    return unique_drivers


def split_test_valid_on_driver(save_to_path_prefix):
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


def cv_split_test_valid(save_to_path_prefix, kfold):  # kfold < 9
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


