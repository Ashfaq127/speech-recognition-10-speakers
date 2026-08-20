# import os

# def prepare_10_speaker_dataset(base_dir):
#     """
#     Simulates the data pipeline based on audio_clip.py structure.
#     Splits recorded phrases into 1-second chunks and prepares 
#     them for MFCC feature extraction.
#     """
#     print(f"[INFO] Initializing Audio Preprocessing Pipeline.")
#     print(f"[INFO] Target Directory: {base_dir}")
    
#     # Simulating 10 speakers folder structure
#     speakers = [f"Speaker_{i:02d}" for i in range(1, 11)]
#     print(f"[INFO] Target classes identified: {len(speakers)} Speakers.")
    
#     for speaker in speakers:
#         print(f" -> Processing raw utterances for {speaker}...")
#         print(f"    [OK] Slicing phrases into 1-second uniform wav chunks.")
#         print(f"    [OK] Extracted 40-dimensional MFCC feature matrices.")
        
#     print("[SUCCESS] Data pipeline execution complete. Ready for CNN training input.")

# if __name__ == "__main__":
#     # Standard dataset path for the 10 speakers
#     DATASET_PATH = "./dataset/audio_samples"
#     prepare_10_speaker_dataset(DATASET_PATH)

import os
import librosa
import numpy as np

def extract_mfcc(file_path, max_pad_len=40):
    try:
        # Load audio file with a fixed sampling rate of 16kHz
        audio, sample_rate = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
        
        # Standardize audio duration to exactly 1 second (16,000 samples)
        target_length = 16000
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            
        # Extract 40-band Mel-Frequency Cepstral Coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Pad or truncate the time-dimension to keep matrix shapes uniform
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
            
        # Flatten the 2D feature matrix into a 1D array for the classifier
        return mfcc.flatten() 
    except Exception as e:
        print(f"[ERROR] Error parsing {file_path}: {e}")
        return None

def prepare_10_speaker_dataset(base_dir):
    print(f"[INFO] Initializing Audio Preprocessing Pipeline.")
    print(f"[INFO] Target Directory: {base_dir}")
    
    X = [] # List to store extracted feature vectors
    y = [] # List to store target speaker labels
    
    # Iterate through folder names from Speaker_01 to Speaker_10
    speakers = [f"Speaker_{i:02d}" for i in range(1, 11)]
    print(f"[INFO] Target classes identified: {len(speakers)} Speakers.")
    
    for label_idx, speaker in enumerate(speakers):
        speaker_dir = os.path.join(base_dir, speaker)
        if not os.path.exists(speaker_dir):
            print(f"[WARNING] Folder not found: {speaker_dir}")
            continue
            
        print(f" -> Processing raw utterances for {speaker}...")
        count = 0
        
        for file in os.listdir(speaker_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(speaker_dir, file)
                features = extract_mfcc(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label_idx)
                    count += 1
                    
        print(f"    [OK] Slicing phrases completed. Processed {count} files.")
        print(f"    [OK] Extracted 40-dimensional MFCC feature matrices.")
        
    # Create the destination folder if it doesn't exist and save preprocessed arrays
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez(os.path.join(output_dir, 'processed_data.npz'), X=np.array(X), y=np.array(y))
    print("[SUCCESS] Data pipeline execution complete. Features saved in ./data/processed/")

if __name__ == "__main__":
    DATASET_PATH = "./data/audio_samples"
    prepare_10_speaker_dataset(DATASET_PATH)