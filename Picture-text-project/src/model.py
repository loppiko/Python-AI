import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import numpy as np
import pickle

import json

class KeyWordsModel:

    def readData(self, trainPath: str) -> list:
        data = pd.read_csv(trainPath, delimiter=';')
        data.columns = ['keywords', 'sentences']

        self.data = data

    
    def readConfig(self, configPath: str) -> dict:
        with open(configPath) as f:
            data = json.load(f)

        self.config = data
    

    def createTokenizers(self, saveTokenizers: bool = True) -> None:
        self.tokenizer_keywords = Tokenizer()
        self.tokenizer_sentences = Tokenizer()

        self.tokenizer_keywords.fit_on_texts(self.data['keywords'])
        self.tokenizer_sentences.fit_on_texts(self.data['sentences'])


        self.keywords_sequences = self.tokenizer_keywords.texts_to_sequences(self.data['keywords'])
        self.sentences_sequences = self.tokenizer_sentences.texts_to_sequences(self.data['sentences'])

        self.keywords_padded = pad_sequences(self.keywords_sequences, maxlen=self.config["max_len_keywords"], padding='post')
        self.sentences_padded = pad_sequences(self.sentences_sequences, maxlen=self.config["max_len_sentences"], padding='post')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.keywords_padded, self.sentences_padded, test_size=0.2, random_state=42)

        if (saveTokenizers):
            with open('models/tokenizer_keywords.pkl', 'wb') as handle:
                pickle.dump(self.tokenizer_keywords, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('models/tokenizer_sentences.pkl', 'wb') as handle:
                pickle.dump(self.tokenizer_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def createModel(self, encoderPath: str, decoderPath: str) -> None:
        encoder_inputs = Input(shape=(self.config["max_len_keywords"],))
        encoder_embedding = Embedding(input_dim=len(self.tokenizer_keywords.word_index) + 1, output_dim=self.config['embedding_dim'])(encoder_inputs)
        encoder_lstm = LSTM(self.config['latent_dim'], return_state=True)
        _, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(self.config["max_len_sentences"],))
        decoder_embedding = Embedding(input_dim=len(self.tokenizer_sentences.word_index) + 1, output_dim=self.config['embedding_dim'])(decoder_inputs)
        decoder_lstm = LSTM(self.config['latent_dim'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(len(self.tokenizer_sentences.word_index) + 1, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.encoder_model = Model(encoder_inputs, encoder_states)
        self.encoder_model.save(encoderPath)

        decoder_state_input_h = Input(shape=(self.config['latent_dim'],))
        decoder_state_input_c = Input(shape=(self.config['latent_dim'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_outputs = decoder_dense(decoder_lstm_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        self.decoder_model.save(decoderPath)




    def performTraining(self, savePath: str) -> None:
        decoder_target_data = np.zeros((len(self.data), self.config["max_len_sentences"]), dtype='int32')

        for i, seq in enumerate(self.sentences_sequences):
            for t, word in enumerate(seq):
                if t > 0:
                    decoder_target_data[i, t - 1] = word


        self.history = self.model.fit([self.X_train, self.y_train], decoder_target_data, batch_size=64, epochs=10, validation_split=0.2)


        self.model.save(savePath)

        loss, accuracy = self.model.evaluate([self.X_test, self.y_test], self.y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")


    
    def loadModel(self, modelPath: str, tokenizerKeyWordPath: str, tokenizerSentencePath: str, configPath: str, encoderPath: str, decoderPath: str) -> None:
        with open(tokenizerKeyWordPath, 'rb') as handle:
            self.tokenizer_keywords = pickle.load(handle)

        with open(tokenizerSentencePath, 'rb') as handle:
            self.tokenizer_sentences = pickle.load(handle)

        self.model = load_model(modelPath)
        self.encoder_model = load_model(encoderPath)
        self.decoder_model = load_model(decoderPath)
        
        self.config = self.readConfig(configPath)



    def generateSequence(self, inputSequence: str) -> None:
        self.readConfig(configPath)

        inputKeywords = self.tokenizer_keywords.texts_to_sequences([inputSequence])
        input_keywords_padded = pad_sequences(inputKeywords, maxlen=self.config['max_len_keywords'], padding='post')

        states_value = self.encoder_model.predict(input_keywords_padded)

        stop_condition = False
        decoded_sentence = ''

        target_seq = np.zeros((1, 1))


        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.tokenizer_sentences.index_word.get(sampled_token_index, '')

            if sampled_word == '<end>' or len(decoded_sentence) > self.config['max_len_sentences']:
                stop_condition = True
            else:
                decoded_sentence += ' ' + sampled_word

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            states_value = [h, c]

        return decoded_sentence.strip()
    

    def saveTestResults(self, resultsDir: str) -> None:

        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.savefig(f"{resultsDir}lost.png")

        plt.plot(self.history.history['accuracy'], label='train_accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig(f"{resultsDir}accuracy.png")





if (__name__ == "__main__"):
    trainPath = "savedData/train.csv"
    savePath = "models/keywords_model.keras"
    configPath = "models/config.json"
    encoderPath = "models/encoder.keras"
    decoderPath = "models/decoder.keras"
    resultsDir = "testResults/LTSM/"

    tokenizer_keyword_path = "models/tokenizer_keywords.pkl"
    tokenizer_sentence_path = "models/tokenizer_sentences.pkl"

    myModel = KeyWordsModel()

    # myModel.readConfig(configPath)
    # myModel.readData(trainPath)
    # myModel.createTokenizers()
    # myModel.createModel(encoderPath, decoderPath)
    # myModel.performTraining(savePath)
    # myModel.saveTestResults(resultsDir)

    myModel.readConfig(configPath)
    myModel.loadModel(savePath, tokenizer_keyword_path, tokenizer_sentence_path, configPath, encoderPath, decoderPath)
    print(myModel.generateSequence("parachute small person small surfboard dimgray"))