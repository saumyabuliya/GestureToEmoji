#dependencies
import numpy as np
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import keras.backend as R

#loading the dataset

data = pd.read_csv("train_set.csv")
dataset = np.array(data)
np.random.shuffle(dataset)
X = dataset
Y = dataset

X = X[:, 1:2501]
Y = Y[:, 0]

#training and test sets

X_train = X[0:11000, :]
X_train = X_train/255
X_test = X[11000:12201, :]
X_test = X_test/255

#taking the transpose of the column vector Y
Y = Y.reshape(Y.shape[0],1)
Y_train = Y[0:11000, :]
Y_train = Y_train.T
Y_test = Y[11000:12201, :]
Y_test = Y_test.T

image_x, image_y = 50, 50

#we apply the one hot encoding technique to change the categorical data
train_y = np_utils.to_categorical(Y_train)
test_y = np_utils.to_categorical(Y_test)
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
#Converting into 50X50 matrix
X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)


#model function
def keras_model(image_x, image_y):
    num_of_classes = 11
    model = Sequential()
    model.add(Conv2D(32, (5,5), input_shape = (image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    model.add(Conv2D(64, (5,5), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(5,5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    filepath = "emojiClassifier.h5"
    checkPoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkPoint1]

    return model, callbacks_list


model, callback_list = keras_model(image_x, image_y)
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs = 1, batch_size = 64, callbacks = callback_list)
scores = model.evaluate(X_test, test_y, verbose = 0)
print("CNN Error: %.2f%%" %(100 - scores[1]*100))
print_summary(model)

model.save("emojiClassifier.h5")