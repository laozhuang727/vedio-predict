# refrence: https://www.jiqizhixin.com/articles/where-cnn-is-looking-grad-cam
# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input

img_path = "img/6.png"
image_size = 224

# Load pre-trained Keras model and the image to classify
# weight_file_dir = '/home/ryan/.keras/models/vgg16_weights_th_dim_ordering_th_kernels.h5'
model = tf.keras.applications.vgg16.VGG16()

# 我们先初始化模型并通过命令行参数加载图片。VGG 网络只接受 (224×224×3) 大小的图片，所以我们要把图片放缩到指定大小。
# 由于我们只通过网络传递一个图像，因此需要扩展第一个维度，将其扩展为一个大小为 1 的批量。
# 然后，我们通过辅助函数 preprocess_input 从输入图像中减去平均 RGB 值来实现图像的归一化。值来实现图像的归一化

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 此处，我们来看看顶部预测的特征图。所以我们得到图像的预测，并给得分靠前的类做个索引。
# 请记住，我们可以为任意类计算特征图。然后，我们可以取出 VGG16 中最后一个卷积层的输出 block5_conv3。得到的特征图大小应该是 14×14×512。
preds = model.predict(x)
class_idx = np.argmax(preds[0])
class_output = model.output[:, class_idx]
last_conv_layer = model.get_layer("block5_conv3")

# 我们计算相类输出值关于特征图的梯度。然后，我们沿着除了通道维度之外的轴对梯度进行池化操作。最后，我们用计算出的梯度值对输出特征图加权。
grads = K.gradients(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

#  然后，我们沿着通道维度对加权的特征图求均值，从而得到大小为 14*14 的热力图。最后，我们对热力图进行归一化处理，以使其值在 0 和 1 之间。
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# 最后，我们使用 OpenCV 来读图片，将获取的热力图放缩到原图大小。我们将原图和热力图混合，以将热力图叠加到图像上。

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
cv2.imshow("Original", img)
cv2.imshow("GradCam", superimposed_img)
cv2.waitKey(0)