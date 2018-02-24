# -*- coding: utf-8 -*-

import keras
from keras.optimizers import *

from keras.models import *
from keras.layers import *


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
    nb_classes = 10
    base_model = Pretrained_model(weights='imagenet', include_top=False,
                                  input_shape=(img_rows, img_cols, 3), classes=nb_classes)

    x = top_classifier(input_tensor=base_model.output, add_gap=True, compile_model=False)
    model = Model(inputs=base_model.input, outputs=x)

    start_layer_no = len(base_model.layers)
    print('load top weights : ', top_classifier_weights_path + '.hdf5')
    f = h5py.File(top_classifier_weights_path + '.hdf5')
    topology.load_weights_from_hdf5_group(f=f, layers=model.layers[start_layer_no:])

    # fine-tuning
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