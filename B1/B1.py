import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, MaxPooling2D, Dense, Conv2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

# Define key parameters to compile the CNN
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

# Create hte CNN model 
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(3, 64, 64)))
    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation = 'softmax'))
    

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model

# Train the model and instatiate early stopping class
def train_model(model, X, y):
	nb_epoch = 100
	batch_size = 15
	early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)  
	history = model.fit(X, y, batch_size=batch_size, epochs=nb_epoch, validation_split=0.25, verbose=0, shuffle=True, callbacks=[early_stopping])
	accuracy = np.mean(history.history['acc'])
	return accuracy, model

# Test and evaluate the model
def test_model(model, X, y):
	return model.evaluate(X, y, verbose=1)[1]






