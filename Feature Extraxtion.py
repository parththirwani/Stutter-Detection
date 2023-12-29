#Feature Extraction
audio_file = "/kaggle/input/stutter-dataset/Vansh_Prolongation_disorder.wav"
y, sr = librosa.load(audio_file)
print(y)
print(sr)


# Calculate RMS energy
energy = librosa.feature.rms(y=y)
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 7))
librosa.display.waveshow(y, sr=sr)
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Create a spectrogram
plt.figure(figsize=(20, 7))
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.show()


plt.plot(energy[0])
# Define x and y coordinates
x = []
y=[]
for i in range(140):
    y.append(0.0025)
for i in range(140):
    x.append(i)
# Create a line plot
plt.plot(x, y)
# plt.scatter(90,0.09)
plt.xlabel("Frame")
plt.ylabel("Energy")


plt.show()
threshold = 0.02  # Adjust this value based on your data
