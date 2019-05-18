#!/usr/bin/env python
# coding: utf-8

# from keras.preprocessing.image import ImageDataGenerator
# 
# 
# # Visualizar una imagen 
# 
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         #shear_range=0.2,
#         zoom_range=[1, 0.9],
#         rotation_range = 10, 
#         horizontal_flip=True)
# 
# train_generator = train_datagen.flow_from_directory(
#         'C:/Users/alexi/Downloads/dogscats/sample',
#         target_size=(150, 150),
#         batch_size=12,
#         shuffle=False,
#         class_mode='binary')
# 
# from matplotlib import pyplot as plt
# %matplotlib inline 
# 
# for i, data in enumerate(train_generator):
#     images = data[0]
#     labels = data[1]
#     print(data[0].shape)
#     print(data[1].shape)
#     print(data[1])
#     for image, label in zip(images, labels): 
#         plt.imshow(image, interpolation='nearest')
#         plt.show()
#         print(label)
#     if i>=0:
#         break

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras
from time import time
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



# DATA SOURCE --------------------------------------------------

batch_size = 19

#C:\Users\alexi\Downloads\dogscats\train
train_data_dir = 'C:/Users/alexi/Downloads/dogs_and_cats/train'
validation_data_dir ='C:/Users/alexi/Downloads/dogs_and_cats/validation'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# MODEL --------------------------------------------------

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# TRAINING --------------------------------------------------

epochs = 15


h = model.fit_generator(
        train_generator,
        steps_per_epoch=300,
        epochs=epochs, 
        validation_data=validation_generator,
        validation_steps=800
)



# Plot training & validation accuracy values
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()





# # RED NEURONAL - TRANSFER LEARNING

# In[2]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# dimesiones para nuestra imagen
img_width, img_height = 150, 150

# Pesos tomados para la VGG16
top_model_weights_path = 'bottleneck_fc_model.h5'

#C:\Users\alexi\Downloads\dogscats\train
train_data_dir = 'C:/Users/alexi/Downloads/dogs_and_cats/train'
validation_data_dir ='C:/Users/alexi/Downloads/dogs_and_cats/validation'

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 30


# DATA SOURCE 

batch_size = 22

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=[1, 0.9],
    horizontal_flip=True)


#Generacion para la parte TRAIN carpeta
generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)


# Aplicando el Transfering learning de VGG16
conv_model = applications.VGG16(include_top=False, weights='imagenet')

# Generacion de las salidas de la red conv seleccionada
bottleneck_features_train = conv_model.predict_generator(
    generator, nb_train_samples // batch_size)

np.save(open('bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)


# Generación para la carpeta de validation
generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_validation = conv_model.predict_generator(
    generator, nb_validation_samples // batch_size)

np.save(open('bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)

print("Train feature maps shape:", bottleneck_features_train.shape)
print("Validation feature maps shape:", bottleneck_features_validation.shape)



train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
train_labels = np.array(
    [0] * int(train_data.shape[0] / 2) + [1] * int(train_data.shape[0] / 2))

validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array(
    [0] * int(validation_data.shape[0] / 2) + [1] * int(validation_data.shape[0] / 2))

#Modelo que quiero tener con las capas
model = Sequential()
model.add(Flatten()
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

#Entrenamiento de nuestra red neuronal

h=model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))


# Plot training & validation accuracy values
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:




