# -*- coding: utf-8 -*-

import glob
import cv2

from keras.models import *
from keras.callbacks import *
import matplotlib.pyplot as plt

import common_model

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

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    picpath = os.path.join('cache', 'loss_' + modelStr + '.png')
    plt.savefig(picpath)
    plt.close(accfig)  # 关闭图


def visualize_class_activation_map(Pretrained_model, model_path, final_conv_layer_name, img_path, img_size, target_class,
                                   preprocess_input=None, show_img=False):
    # model = load_model(model_path)
    # 加载模型数据和weights
    # model = model_from_json(open(model_path + '.json').read())
    # model.load_weights(model_path + '.h5')
    base_model = Pretrained_model(weights='imagenet', include_top=False,
                                  input_shape=(img_size[0], img_size[1], 3), classes=10)
    x = common_model.top_classifier(input_tensor=base_model.output, add_gap=True, compile_model=False)
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

