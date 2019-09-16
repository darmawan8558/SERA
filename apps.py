from flask import Flask, render_template
import flask
import numpy
import numpy as np
import pandas as pd
import csv
import glob, os
import time
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import logfbank
from scipy.io.wavfile import read
from pylab import *
from scipy.fftpack import dct
import sys
import scipy.io.wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import pickle

from scipy.io.wavfile import write
from playsound import playsound

plt.close('all')

app = Flask(__name__) #panggil modul flask

fs = 44100; #sample rate (hz)
d = 3; #duration recording
		

@app.route('/')
@app.route('/index')


def index():
	return flask.render_template('seravdua.html')
	
@app.route("/rec/" , methods=["GET", "POST"])
def rec():
	myrecord = sd.rec(int(d*fs), fs, 1,blocking = True)
	wav_file_out = 'static/output.wav'
	write(wav_file_out, fs, myrecord)
	msg = " Hore, suaramu telah direkam."
		# path untuk koneksi ke direktori data suara
	path_1 = 'static/'
	# path untuk Simpan ciri ke file ".CSV"
	path_csv_1 = 'static/csvtest_1.csv'
	csv_file = open(path_csv_1, "w")
	time_start = time.time()
	print('[INFO]: Ekstrasi Ciri dimulai!')
	print('-------------------------------------------------------------------------------------------')

	with open(path_csv_1, mode='w') as csv_file:
		#memberi nama baris di csv 
		fieldnames = ['Katagori', 'mel_1', 'mel_2', 'mel_3', 'mel_4', 'mel_5', 'mel_6', 'mel_7', 
					  'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12', 'mel_13', 'Katagori_Index']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames) 
		writer.writeheader()
		files = []

		files = [filename for filename in glob.glob(path_1 + "**/*.wav", recursive=True)]
		for filename in files:
			print('[EKSTRAK]: ',filename)

			(rate,sig) = wav.read(filename)
			mfcc_var = mfcc(sig, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
			mfcc_var = mfcc_var[1:2,:]
			mfcc_var = mfcc_var.astype(int)
			
			#Pengelompokan emosi per data suara pada file CSV

			label = "Unknown, "
			label_index = ", 0"


			#Konversi hasil disimpan ke file CSV
			mfcc1=numpy.array(mfcc_var.flatten())
			a_str=','.join(str(x) for x in mfcc1)
			csv_file.write(label + a_str + label_index + '\n')
	time_end = time.time()

	print('-------------------------------------------------------------------------------------------')
	print('[INFO] Ektraksi Ciri Selesai')
	print('[INFO] Lama Ekstraksi: ',round(time_end - time_start,3),' Second')

	csv_file.close()
	DTest_1 = pd.read_csv('static/csvtest_1.csv')
	XT1 = DTest_1.iloc[:,1:13]
	model = pickle.load(open("model.pkl","rb"))
	predictions_1 = model.predict(XT1)
	return flask.render_template('seravdua.html', msg=msg, hasil = "Emosimu saat ini : " + predictions_1[0] + ".")
	
@app.route("/play/" , methods=["GET", "POST"])
def play():
	playsound('static/output.wav')
	return flask.render_template('seravdua.html')
	
if __name__ == '__main__': #RUN PROGRAM
	app.run() #LOCALHOST, 
	
