import numpy as np
import time
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

def main():
    model = Sequential() 
    model.add(Conv2D(64, kernel_size = (3,3), strides = (1,1), activation= 'relu', input_shape = (224,224, 1),  padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size = (3,3), strides = (1,1), activation= 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    
    model.add(Conv2D(128, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))  
    
    model.add(Conv2D(256, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(256, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(256, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    
    model.add(Conv2D(512, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(512, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(512, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    
    model.add(Conv2D(512, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(512, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(512, kernel_size = (3, 3), strides = (1,1), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(64, activation = 'relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(32, activation = 'relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(4, activation = 'relu'))
    model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = .0001))
    return model

X_train = np.load('X_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
X_test = np.load('X_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)
name = 'Nutrition_detection-{}'.format(int(time.time()))
tensor_board = TensorBoard(log_dir='logs/{}'.format(name))
# early_stopping = EarlyStopping(monitor='val_loss', verbose = 1, patience = 50, min_delta = .00075)
# model_checkpoint = ModelCheckpoint('ModelWeights.h5', verbose = 1, save_best_only = True,
#                                   monitor = 'val_loss')
# lr_plat = ReduceLROnPlateau(patience = 2, mode = 'min')
epochs = 1
batch_size = 5
model = main()
model_history = model.fit(X_train, y_train, validation_split = 0.1, batch_size = batch_size,
            epochs = epochs,
     callbacks = [ tensor_board], verbose = 1)
# early_stopping, model_checkpoint, lr_plat,
model.save('object_detection_model4.h5')
main()