#import librosa.display
import numpy as np
import pickle
import audiomentations as A
import soundfile as sf
import os
import librosa
import warnings
import pyaudio
import wave


from time import sleep





OUTPUT_FILENAME = "audio_output.wav"
# Your code that produces the warning goes here

# Filter out the specific warning by matching the warning message
warnings.filterwarnings("ignore", category=UserWarning, message="Warning: input samples dtype is np.float64. Converting to np.float32")



with open('rfc.pkl' , 'rb') as f:
    loaded_model = pickle.load(f)

num_mfcc_coefficients = 15
desired_shape = (1, num_mfcc_coefficients)

def Test_preprocess_audio(audio_file):
        
        Test_preprocess_data = []
        original_audio, sr = sf.read(audio_file)

        # Apply augmentation to create augmented audio
        augment1 = A.AddGaussianNoise(p=0.2)
        augment2 = A.TimeStretch(p=0.2)
        augment3 = A.PitchShift(p=0.2)
        augment4 = A.Shift(p=0.2)
        augment5 = A.TimeMask(p=0.2)

        augmented_audio1 = augment1(samples=original_audio, sample_rate=sr)
        augmented_audio2 = augment2(samples=original_audio, sample_rate=sr)
        augmented_audio3 = augment3(samples=original_audio, sample_rate=sr)
        augmented_audio4 = augment4(samples=original_audio, sample_rate=sr)
        augmented_audio5 = augment5(samples=original_audio, sample_rate=sr)

        # Perform feature extraction (e.g., MFCCs) on original and augmented audio
        temp_audio = []
        for audio in [original_audio, augmented_audio1, augmented_audio2, augmented_audio3, augmented_audio4, augmented_audio5]:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc_coefficients)

            # Normalize the MFCCs (optional but recommended)
            mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

            # # Reshape or pad the MFCCs to match the desired input shape
            num_frames = mfccs.shape[1]
            if num_frames < desired_shape[0]:
                mfccs = np.pad(mfccs, ((0, 0), (0, desired_shape[0] - num_frames)), mode='constant')
            elif num_frames > desired_shape[0]:
                mfccs = mfccs[:, :desired_shape[0]]

            # Append the preprocessed data and label
            # Test_preprocess_data.append(mfccs.T)  # Transpose the data
            temp_audio.append(mfccs.T)
        # Append the preprocessed data and label
        new_temp= []
        for audio in temp_audio:
            for aud in audio.tolist():
                for au in aud:
                    new_temp.append(au)
                
        # print("TA", len(new_temp),new_temp)
        Test_preprocess_data.append(new_temp)  # Transpose the data
            
        # Stack the preprocessed data into a 3D array
        X = np.array(Test_preprocess_data, dtype=np.float32)#.reshape(-1,13)

        return X

def Predict_Label(audio_file):
    processed_data = (Test_preprocess_audio(audio_file))
    # print(processed_data.shape)
    y_pred=loaded_model.predict(processed_data)
    # print("Y_pred", y_pred)
    # y_pred = np.bincount(y_pred).argmax()
    y_pred = y_pred[0]
    # print(y_pred)
    # y_pred = int(np.median(y_pred))
    # print(y_pred)
    if y_pred == 0:
        return('belly_pain')
    if y_pred == 1:
        return('burping')
    if y_pred == 2:
        return('discomfort')
    if y_pred == 3:
        return('hungry')
    if y_pred == 4:
        return('tired')
    return "Not detected"

def RecordAudio(dev,duration=5):
     FORMAT = pyaudio.paInt16
     CHANNELS = dev
     RATE = 44100  # Sample rate (samples per second)
     CHUNK = 1024  # Size of each audio chunk
     RECORD_SECONDS = duration
     audio = pyaudio.PyAudio()
     stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
     print("Recording...")
     frames = []

# Record audio for the specified duration
     for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

     print("Finished recording.")
     stream.stop_stream()
     stream.close()
     audio.terminate()

# Save the recorded audio to a WAV file
     with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

     print(f"Audio saved to {OUTPUT_FILENAME}")


if __name__=="__main__":

    #RecordAudio(1)
    print(f"Predicting..            ")
    outP = Predict_Label(OUTPUT_FILENAME)
    print(outP)

