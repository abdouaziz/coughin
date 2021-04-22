import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050


class _Keyword_Spotting_Service:

    model = None
    _mapping = [
        "neg",
        "pos",

    ]
    _instance = None

    def predict(self, file_path):

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                     hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(
            SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

  
    assert kss is kss1

    # make a prediction
    keyword = kss.predict("a.mp3")
    print(keyword)
