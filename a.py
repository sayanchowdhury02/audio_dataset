import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Paths
input_root = 'dataset'
output_root = 'spectrogram_dataset'

# Create output folder if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Loop through all .wav files in the dataset
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, input_root)
            
            # Prepare output path
            output_path = os.path.join(output_root, relative_path)
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            # Change extension to .png
            output_image_path = os.path.splitext(output_path)[0] + '.png'

            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=None)

                # Create mel spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)

                # Plot and save image
                plt.figure(figsize=(2.56, 2.56), dpi=100)
                librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()

                print(f"Saved: {output_image_path}")

            except Exception as e:
                print(f"Failed on {file_path}: {e}")
