import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History, ModelCheckpoint

from tensorflow.math import confusion_matrix


class SceneClassifier():


    def createModel(self) -> None:
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(*self.imageSize, 3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.listOfCategories), activation='softmax'))

        opt = Adam(learning_rate=0.001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        
    
    def summarize_diagnostics(self, cm: list, saveResultsDir: str) -> None:
        # plot loss
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f"{saveResultsDir}confusion_matrix.png")

        # Plotting training and validation accuracy
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', color='grey')
        plt.legend()

        # Plotting training and validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', color='grey')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{saveResultsDir}accuracy.png")

        plt.close()


    def readDataset(self, datasetPath: str) -> None:
        self.listOfCategories = sorted([d for d in os.listdir(datasetPath) if os.path.isdir(os.path.join(datasetPath, d))])

        print(self.listOfCategories)

        self.train_dataset = image_dataset_from_directory(
            datasetPath,
            validation_split=0.2,
            subset="training",
            seed=288497,
            image_size=self.imageSize,
            batch_size=self.batchSize
        )

        self.validation_dataset = image_dataset_from_directory(
            datasetPath,
            validation_split=0.2,
            subset="validation",
            seed=288497,
            image_size=self.imageSize,
            batch_size=self.batchSize
        )

        AUTOTUNE = tensorflow.data.AUTOTUNE

        self.train_dataset = self.train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.validation_dataset = self.validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)




    def calculate_confusion_matrix(self):
        true_labels = []
        predicted_labels = []

        for x, y_true in self.validation_dataset:
            y_pred = self.model.predict(x)
            true_labels.extend(tensorflow.argmax(y_true, axis=1).numpy())
            predicted_labels.extend(tensorflow.argmax(y_pred, axis=1).numpy())
        
        cm = confusion_matrix(true_labels, predicted_labels)
        return cm


    def performTraining(self, datasetPath: str, imageSize: tuple[int, int], batchSize: int, epochs, saveDir: str, saveResultsDir: str) -> None:
        self.imageSize = imageSize
        self.batchSize = batchSize
        self.history = History()
        self.checkpoint = ModelCheckpoint(
            filepath=f"{saveDir}sceneClassifier.keras",
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        

        self.readDataset(datasetPath)
        self.createModel()


        self.model.fit(self.train_dataset, validation_data=self.validation_dataset, epochs=epochs, batch_size=self.batchSize, callbacks=[self.history, self.checkpoint])

        # cm = self.calculate_confusion_matrix()

        # self.summarize_diagnostics(cm, saveResultsDir)

    

    def performClassification(self): 
        pass


if (__name__ == "__main__"):
    datasetPath = "datasets/scenes/"
    saveResultsDir = "testResults/scenes/"
    saveDir = "models/"
    imageSize = (50, 50)
    batchSize = 32
    epochs = 1

    model = SceneClassifier()
    model.performTraining(datasetPath, imageSize, batchSize, epochs, saveDir, saveResultsDir)