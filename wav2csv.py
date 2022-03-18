
import scipy.io.wavfile as wav
import os
from python_speech_features import mfcc
from python_speech_features import logfbank
import numpy

directoryName = "./dataset/wav/noise"
resultsDirectory = "./dataset/csv/noise"
for filename in os.listdir(directoryName):
    if filename.endswith('.wav'): # only get MFCCs from .wavs

        (rate,sig) = wav.read(directoryName + "/" +filename)


        mfcc_feat = mfcc(sig,rate)

        fbank_feat = logfbank(sig,rate)


        outputFile = resultsDirectory + "/" + os.path.splitext(filename)[0] + ".csv"
        file = open(outputFile, 'w+')
        numpy.savetxt(file, mfcc_feat, delimiter=",")
        file.close() # close file