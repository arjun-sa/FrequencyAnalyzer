import librosa
import numpy as np
import pandas as pd
import os

BAD_RANGE = (20, 40)
audio_folder = "/Users/arjunsakthi/Downloads/sesh3"
results = []

def analyze_audio(file_path, bad_range):
    y, sr = librosa.load(file_path, sr=None)
    stft = librosa.stft(y)
    spectrogram = np.abs(stft)**2
    frequencies = librosa.fft_frequencies(sr=sr)
    total_energy = np.sum(spectrogram)
    bad_indices = np.where((frequencies >= bad_range[0]) & (frequencies <= bad_range[1]))[0]
    bad_energy = np.sum(spectrogram[bad_indices, :])
    bad_percentage = (bad_energy / total_energy) * 100
    return bad_percentage

for file_name in os.listdir(audio_folder):
    if file_name.endswith(('.mp3', '.wav')):
        file_path = os.path.join(audio_folder, file_name)
        bad_percentage = analyze_audio(file_path, BAD_RANGE)
        results.append({"File Name": file_name, "Bad Frequency %": bad_percentage})

name = audio_folder[29:34]

df = pd.DataFrame(results)
df = df.sort_values(by="Bad Frequency %", ascending=True)
df.to_csv(name + "_frequency_analysis_results.csv", index=False)
print("Analysis complete. Results saved to " + name + "_frequency_analysis_results.csv.")