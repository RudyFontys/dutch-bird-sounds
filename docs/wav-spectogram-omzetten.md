
# .wav-clips omzetten naar .npy Spectrogrammen

Hieronder staat het vervolgscript dat de dataset_v1.csv uitleest, elke .wav-clip omzet naar een gestandaardiseerd Log-Mel Spectrogram (shape: 128 x 173), en de CSV bijwerkt met het pad naar de nieuwe .npy-bestanden.

Voer dit script uit nadat je de .wav-bestanden hebt gegenereerd (convert.py):

```python
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
```


## Mogelijke stap 2: Direct inladen in PyTorch of TensorFlow

Omdat elke array exact de vorm (128, 173) heeft, kun je ze direct in een 2D Convolutional Neural Network (CNN) stoppen (net als een zwart-witafbeelding met 1 kanaal):

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class VogelDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec = np.load(self.df.iloc[idx]["spec_pad"])
        # Voeg kanaal-dimensie toe -> (1, 128, 173)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.df.iloc[idx]["label_id"], dtype=torch.long)
        return spec_tensor, label
```
## Optie B: TensorFlow / Keras tf.data.Dataset

```python
import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv("vogel_dataset_v1/dataset_v1.csv")

# Laad direct in geheugen (125 items is maar ~11 MB)
X = np.array([np.load(p) for p in df["spec_pad"]])[..., np.newaxis] # Shape: (125, 128, 173, 1)
y = df["label_id"].values
```

dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(125).batch(16)


## Bruikbaar voor V1 met 125 clips

Met 25 voorbeelden per soort is de dataset compact. Gebruik bij het trainen *SpecAugment* (willekeurig horizontale of verticale balkjes maskeren op je spectrogram) om overfitting te voorkomen en het model robuuster te maken tegen achtergrondruis.

## Uitleg van het begrip "overfitting" en voorlopige keuze Data Augmentation

Overfitting betekent dat je AI-model de trainingsdata letterlijk uit zijn hoofd leert in plaats van de algemene kenmerken van het vogelgeluid te herkennen. Hierdoor scoort het model vrijwel 100% op de opnames waarmee het is getraind, maar presteert het slecht op nieuwe, ongehoorde vogelgeluiden.

**Hoe werkt dat bij de compacte Xeno-Canto vogeldataset?**

Met 25 opnames per soort is de **kans op overfitting groot**. In plaats van te leren "hoe een merel klinkt", leert het model bijvoorbeeld:
1. "Als er op seconde 2 een klikje of windruis zit, dan is het een merel."
2. "Deze specifieke opname heeft een bepaalde achtergrondruis uit een bos in Duitsland, dus dit is een roodborst."
Zodra men daarna buiten een nieuwe merel opneemt (zonder die specifieke achtergrondruis), herkent het model de vogel niet meer.

**Hoe herken je overfitting?**

Tijdens het trainen splits je de dataset in een **trainingsset** (bijv. 80% = 100 clips) en een **validatieset** (bijv. 20% = 25 clips die het model niet gebruikt om van te leren, alleen om zichzelf te testen):
1. **Gezonde training**: De foutmarge (loss) daalt op zowel de trainingsset als de validatieset.
2. **Overfitting**: De nauwkeurigheid op de trainingsset blijft stijgen naar 99%, maar op de validatieset stagneert deze (bijv. op 65%) of wordt zelfs slechter.

**3 mogelijke manieren om dit in V1 te voorkomen**

1. **Data Augmentation (SpecAugment)**: Willekeurige stukjes van het spectrogram afdekken of de audio licht verschuiven in toonhoogte/tijd, zodat geen enkele opname twee keer exact hetzelfde is.
2. **Dropout toevoegen**: Tijdens het trainen steeds willekeurig een deel (bijv. 30%) van de neuronen in je netwerk uitzetten, zodat het model niet afhankelijk wordt van een specifiek detail.
3. **Transfer Learning gebruiken**: Een bestaand, voorgetraind beeld- of audiomodel (zoals ResNet of MobileNet) als basis gebruiken in plaats van helemaal vanaf nul te beginnen.l opneemt (zonder die specifieke achtergrondruis), herkent het model de vogel niet meer.

In mijn plan van aanpak is al een keuze gemaakt voor **Data Augmentation**. Als de resultatien hiervan tegenvallen test ik een andere manier.