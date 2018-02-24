# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import glob
import keras
from keras.optimizers import *

from math import ceil

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.callbacks import *

import common_model as cm
import visulization
import data_process as DP

np.random.seed(2017)
data_total_batch = 10
color_type_global = 3


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
        = DP.split_test_valid_on_driver(modelStr)

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

    train_data_generator = DP.data_generator_from_file(path=train_cvs_path, batch_size=batch_size,
                                                    img_rows=img_rows, img_cols=img_cols,
                                                    isvalidation=True,
                                                    preprocess=preprocess)
    bottleneck_train_y = model.predict_generator(generator=train_data_generator,
                                                 steps=ceil(train_number / batch_size),
                                                 verbose=1)
    # Get valid bottleneck features
    valid_data_generator = DP.data_generator_from_file(path=valid_cvs_path, batch_size=batch_size,
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
    top_model = cm.top_classifier(input_shape=model.output_shape[1:], add_gap=False, compile_model=True)

    bf_y_train = DP.get_y_from_cvs_file(train_cvs_path)
    bf_y_valid = DP.get_y_from_cvs_file(valid_cvs_path)

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
    visulization.draw_training_result(history=hist, modelStr=modelStr)


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
        = DP.cv_split_test_valid(modelStr, kfold)

    test_samples = DP.get_test_numbers()
    yfull_predict = np.zeros(shape=(test_samples, 10), dtype=np.float)

    for i in range(kfold):
        train_number = train_img_len_list[i]
        valid_number = valid_img_len_list[i]
        print('Split train: ', train_number)
        print('Split valid: ', valid_number)

        train_cvs_path = train_cvs_path_list[i]
        valid_cvs_path = valid_cvs_path_list[i]

        train_data_generator = DP.data_generator_from_file(train_cvs_path, batch_size=batch_size,
                                                           img_rows=img_rows, img_cols=img_cols,
                                                           isvalidation=False,
                                                           preprocess=preprocess)
        valid_data_generator = DP.data_generator_from_file(valid_cvs_path, batch_size=batch_size,
                                                           img_rows=img_rows, img_cols=img_cols,
                                                           isvalidation=True,
                                                           preprocess=preprocess)

        model = cm.fine_tuned_model(Pretrained_model=MODEL, freeze_layers_num=freeze_layers_num,
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
        print('steps_per_epoch: ', ceil(train_number * (DP.da_multiple + 1) / batch_size))
        print('validation_steps: ', ceil(valid_number / batch_size))
        hist = model.fit_generator(
            generator=train_data_generator,
            steps_per_epoch=ceil(train_number * (DP.da_multiple + 1) / batch_size),
            epochs=nb_epoch,
            verbose=1,
            callbacks=[MemoryCallback(), tensorboard, earlystop, reduceLROnPlateau, modelCheckpoint],
            # changelearningrate],
            validation_data=valid_data_generator,
            validation_steps=ceil(valid_number / batch_size),
            shuffle=True)

        # 绘制训练过程
        visulization.draw_training_result(history=hist, modelStr=modelStr + '.fold_' + str(i))

        # predict
        model.load_weights(filepath=modelStr + '.fold_' + str(i) + '.hdf5')
        # model.save_weights(filepath=modelStr + '.fold_' + str(i) + '.hdf5')

        print('test_steps_per_epoch: ', ceil(test_samples / batch_size))
        test_data_generator = DP.test_data_generator_from_path(batch_size, img_rows, img_cols,
                                                               preprocess=preprocess)
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
    x = cm.top_classifier(input_tensor=base_model.output, add_gap=True, compile_model=False)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(filepath=modelStr + '.hdf5')

    # predict
    test_samples = DP.get_test_numbers()
    print('test_steps_per_epoch: ', ceil(test_samples / batch_size))
    test_data_generator = DP.test_data_generator_from_path(batch_size, img_rows, img_cols,
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
    test_samples = DP.get_test_numbers()
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

    train_data_generator = DP.data_generator_from_file(path=train_cvs_path, batch_size=batch_size,
                                                       img_rows=img_rows, img_cols=img_cols,
                                                       isvalidation=False,
                                                       preprocess=preprocess)
    bottleneck_train_y = model.predict_generator(generator=train_data_generator,
                                                 steps=ceil(train_number / batch_size),
                                                 verbose=1)
    # Get valid bottleneck features
    valid_data_generator = DP.data_generator_from_file(path=valid_cvs_path, batch_size=batch_size,
                                                       img_rows=img_rows, img_cols=img_cols,
                                                       isvalidation=False,
                                                       preprocess=preprocess)
    bottleneck_valid_y = model.predict_generator(generator=valid_data_generator,
                                                 steps=ceil(valid_number / batch_size),
                                                 verbose=1)

    test_samples = DP.get_test_numbers()
    print('test_steps_per_epoch: ', ceil(test_samples / batch_size))
    test_data_generator = DP.test_data_generator_from_path(batch_size, img_rows, img_cols,
                                                           preprocess=preprocess)
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
        = DP.split_test_valid_on_driver('ensemble_bf_')
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

    y_train = DP.get_y_from_cvs_file(train_cvs_path)
    y_valid = DP.get_y_from_cvs_file(valid_cvs_path)

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
    visulization.draw_training_result(history=hist, modelStr='ensemble_bf')

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

    visulization.show_all_classes_cam(Pretrained_model=VGG16,
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