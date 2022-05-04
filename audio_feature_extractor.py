import pandas as pd
import numpy as np
import librosa, librosa.display #librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.
import pickle

class audio_feature_extractor:
    def __init__(self):
        self.features = {}
        
    def insert_(self,key,value):
        self.features[key] = value
        
    def reset(self):
        self.features = {}
        
    def extract_features(self,signal,sample_rate):
        #signal, sample_rate = librosa.load(file, sr=22050)#plot signal
        self.insert_("signal",signal)
        self.insert_("sample rate",sample_rate)
        # FFT -> power spectrum
        # perform Fourier transform
        fft = np.fft.fft(signal)
        # calculate abs values on complex numbers to get magnitude
        spectrum = np.abs(fft)
        # create frequency variable
        f = np.linspace(0, sample_rate, len(spectrum))
        # take half of the spectrum and frequency
        left_spectrum = spectrum[:int(len(spectrum)/2)]#plot_fft
        left_f = f[:int(len(spectrum)/2)]#plot_fft
        self.insert_("FFT",spectrum)
        # STFT -> spectrogram
        hop_length = 512 # in num. of samples
        frame_length = 2*hop_length
        n_fft = 2048 # window in num. of samples
        # calculate duration hop length and window in seconds
        hop_length_duration = float(hop_length)/sample_rate
        n_fft_duration = float(n_fft)/sample_rate
        #print("STFT hop length duration is: {}s".format(hop_length_duration))
        #print("STFT window duration is: {}s".format(n_fft_duration))
        # perform stft
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        # calculate abs values on complex numbers to get magnitude
        spectrogram = np.abs(stft)#plot stft
        self.insert_("STFT",spectrogram)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        self.insert_("STFT in Decibels",log_spectrogram)
        #ZCR
        n1=0
        n2=len(signal)
        #ZC Calculation
        zero_crossings = librosa.zero_crossings(signal[n1:n2], pad=False)
        self.insert_("Zero Crossings",sum(zero_crossings))
        #ZCR Calculation 
        zcrs = librosa.feature.zero_crossing_rate(signal[n1:n2],frame_length=frame_length,hop_length=hop_length)
        zcrs=list(np.around(np.array(zcrs)*frame_length).astype(int))
        self.insert_("Zero Crossing Rate",zcrs[0])
        # MFCCs
        # extract 13 MFCCs
        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
        self.insert_("13 MFCCs",MFCCs)
        #Calculating Energy
        energy = np.array([sum(abs(signal[i:i+frame_length]**2)) for i in range(0, len(signal), hop_length)])
        self.insert_("Energy",energy)
        #Calculating RMSE
        rmse = librosa.feature.rms(signal, frame_length=frame_length, hop_length=hop_length, center=True)
        self.insert_("RMSE",rmse[0])
        return self.get_features()
        
    def get_features(self):
        return self.features
    
    def get_feature_row(self,path,sample_rate):
        self.extract_features(path,sample_rate)
        features=self.get_features()
        row={}
        fmap={}
        count=1
        for i in features['13 MFCCs']:
            row[str(count)]=[i]
            count+=1
        row['Zero Crossing Rate']=[features['Zero Crossing Rate']]
        row['Energy']=[features['Energy']]
        row['RMSE']=[features['RMSE']]
        self.reset()
        return row
    
class gender_detector():
    def __init__(self):
        pickle_in = open("gender_classifier.pickle","rb")
        self.clf = pickle.load(pickle_in)
    def detect(self,feature_vector=[]):
        feature_vector=[feature_vector]
        return self.clf.predict(feature_vector)