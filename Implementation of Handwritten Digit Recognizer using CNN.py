#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# # In[2]:


X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)


# # In[5]:


sample_shape = X_train[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

# # Reshape data
X_train = X_train.reshape(len(X_train), input_shape[0], input_shape[1], input_shape[2])
X_test  = X_test.reshape(len(X_test), input_shape[0], input_shape[1], input_shape[2])

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Conv2D(32, 
#                  kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape,
#                  strides=1))

# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
#                        strides=2))

# model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, 
#                  activation='relu'))

# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                        strides=2))

# model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, 
#                  activation='relu'))

# model.add(tf.keras.layers.Flatten())

# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# print(model.summary())


# # In[6]:


# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy', 
#              metrics=['accuracy'])


# # In[7]:


# model.fit(X_train, y_train, epochs=5)


# # In[8]:


# val_loss, val_acc = model.evaluate(X_test, y_test)
# print(val_loss, val_acc)


# # In[9]:


# model.save('handwritten-train.model')


# # In[3]:


import numpy as np
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('handwritten-train.model')

predictions = new_model.predict([X_test])

while(True):
    answer = int(input('Insira o numero do item para teste: '))
    print(predictions[answer])
    print(np.argmax(predictions[answer]))
    plt.imshow(X_test[answer])
    plt.show()