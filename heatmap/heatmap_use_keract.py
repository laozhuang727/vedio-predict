# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""
import cv2
import keract
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input

img_path = "img/cat_dog.png"
image_size = 224

model = VGG16()

# url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/250px-Gatto_europeo4.jpg'
# response = requests.get(url)
# origin_image = Image.open(BytesIO(response.content))
# image = origin_image.crop((0, 0, 224, 224))
img = image.load_img(img_path, target_size=(image_size, image_size))


img = img_to_array(img)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
img = preprocess_input(img)
yhat = model.predict(img)
label = decode_predictions(yhat)
label = label[0][0]
print('{} ({})'.format(label[1], label[2] * 100))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
activations = keract.get_activations(model, img, layer_name='block5_conv3')
first = activations.get('block5_conv3')
# keract.display_activations(activations)
# keract.display_heatmaps(activations, input_image=image)



grad_trainable_weights = keract.get_gradients_of_activations(model, img, yhat, layer_name='block5_conv3')

print(grad_trainable_weights['block5_conv3'].shape)
grad_trainable_weights = tf.convert_to_tensor(grad_trainable_weights['block5_conv3'])


pooled_grads = K.mean(grad_trainable_weights, axis=(0, 1, 2))

# 我们计算相类输出值关于特征图的梯度。然后，我们沿着除了通道维度之外的轴对梯度进行池化操作。最后，我们用计算出的梯度值对输出特征图加权。
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, first[0]), axis=-1)

#  然后，我们沿着通道维度对加权的特征图求均值，从而得到大小为 14*14 的热力图。最后，我们对热力图进行归一化处理，以使其值在 0 和 1 之间。
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

print(heatmap.shape)

# 最后，我们使用 OpenCV 来读图片，将获取的热力图放缩到原图大小。我们将原图和热力图混合，以将热力图叠加到图像上。
img = cv2.imread(img_path)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
cv2.imshow("Original", img)
cv2.imshow("GradCam", superimposed_img)
cv2.waitKey(0)