import os
import numpy as np
import pandas as pd
import librosa

OUTPUT_DIR = "vogel_dataset_v1"
SPEC_DIR = os.path.join(OUTPUT_DIR, "spectrograms_npy")
CSV_PAD = os.path.join(OUTPUT_DIR, "dataset_v1.csv")

os.makedirs(SPEC_DIR, exist_ok=True)
df = pd.read_csv(CSV_PAD)

# Spectrogram instellingen (standaard voor audio-classificatie)
N_MELS = 128      # Aantal frequentiebanden (hoogte)
N_FFT = 2048      # Venstergrootte voor FFT
HOP_LENGTH = 512  # Stapgrootte (4 sec bij 22050 Hz geeft exact 173 tijdstappen)

spec_paden = []

for idx, rij in df.iterrows():
    wav_pad = rij["bestandspad"]
    xc_id = rij["xc_id"]
    soort_naam = rij["klasse"].replace(" ", "_")

    # Maak submap per klasse
    klasse_spec_map = os.path.join(SPEC_DIR, soort_naam)
    os.makedirs(klasse_spec_map, exist_ok=True)
    npy_pad = os.path.join(klasse_spec_map, f"XC{xc_id}.npy")

    # 1. Laad de uniforme 4-sec .wav clip
    y, sr = librosa.load(wav_pad, sr=None)

    # 2. Bereken Mel-spectrogram en zet om naar decibels (Log-Mel)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. Normaliseer waarden naar [0, 1] voor stabiele AI-training
    log_mel_norm = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)

    # 4. Sla op als float32 .npy array (shape: 128 x 173)
    np.save(npy_pad, log_mel_norm.astype(np.float32))
    spec_paden.append(npy_pad)

# Voeg het nieuwe pad toe aan de CSV en sla op
df["spec_pad"] = spec_paden
df.to_csv(CSV_PAD, index=False)

print(f"Klaar! {len(df)} spectrogrammen opgeslagen. Shape per item: {log_mel_norm.shape}")