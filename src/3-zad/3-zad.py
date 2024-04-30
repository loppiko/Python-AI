# baseline model for the dogs vs cats dataset
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import History

from tensorflow.math import confusion_matrix


DATASET_DIR = "dataset/"


def define_model(input_shape):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(*input_shape, 3)))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


def calculate_confusion_matrix(model, validation_dataset):
    true_labels = []
    for _, labels in validation_dataset:
        true_labels.extend(labels.numpy())

    predicted_probs = model.predict(validation_dataset)
    predicted_classes = np.where(predicted_probs > 0.5, 1, 0)

    cm = confusion_matrix(true_labels, predicted_classes)
    return cm


# plot diagnostic learning curves
def summarize_diagnostics(history, cm):
	# plot loss
	plt.figure(figsize=(10, 7))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title('Confusion Matrix')
	plt.savefig("confusion_matrix.png")

	# Plotting training and validation accuracy
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.plot(history.history['accuracy'], label='Training Accuracy')
	plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.grid(True, linestyle='--', color='grey')
	plt.legend()

	# Plotting training and validation loss
	plt.subplot(1, 2, 2)
	plt.plot(history.history['loss'], label='Training Loss')
	plt.plot(history.history['val_loss'], label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.grid(True, linestyle='--', color='grey')
	plt.legend()

	plt.tight_layout()
	plt.savefig("accuracy.png")


def run_test_harness(train_dataset, validation_dataset, batch_size, input_shape):    
	
	model = define_model(input_shape)
	
	history = History()

	model.fit(train_dataset, validation_data=validation_dataset, epochs=5, batch_size=batch_size, validation_split=0.2, callbacks=[history])

	cm = calculate_confusion_matrix(model, validation_dataset)

	summarize_diagnostics(history, cm)
     

if (__name__ == "__main__"):
	image_size = (50, 50)
	batch_size = 8

	train_dataset = image_dataset_from_directory(
	  DATASET_DIR,
	  validation_split=0.2,
	  subset="training",
	  seed=288498,
	  image_size=image_size,
	  batch_size=batch_size
	)

	validation_dataset = image_dataset_from_directory(
	  DATASET_DIR,
	  validation_split=0.2,
	  subset="validation",
	  seed=288498,
	  image_size=image_size,
	  batch_size=batch_size
	)

	run_test_harness(train_dataset, validation_dataset, batch_size, image_size)