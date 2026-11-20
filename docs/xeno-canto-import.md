# Uitleg hoe ik van Xeno Canto 125 vogelgeluiden download heb bestaande uit 5 klassen. Dus een dataset met csv.

Je kunt vogelgeluiden het makkelijkst downloaden van Xeno Canto met een klein Python-script via de Xeno-canto API v3. Hiermee haal je per vogelsoort (klasse) 25 opnames op (5 × 25 = 125 bestanden) en sla je alle metadata én bestandspaden direct op in één CSV-bestand.
https://xeno-canto.org/admin.php/explore/api

Let op: Xeno-canto gebruikt API v3 (v2 is offline). Je hebt hiervoor een gratis persoonlijke API-sleutel nodig.
https://xeno-canto.org/article/304

## Stap 1: API-sleutel ophalen

1. Maak een gratis account aan op Xeno-canto en log in.
2. Ga naar je Account-pagina en kopieer je persoonlijke API-sleutel.
https://xeno-canto.org/article/304

## Opm. als je in een virtuele omgeving werkt

Werk je in VS Codium in een .venv virtuele omgeving, switch dan eerst naar de juiste plek:
1. Druk op Ctrl + Shift + P (of F1) om het Command Palette te openen.
2. Typ Python: Select Interpreter en druk op Enter.
3. Kies de optie Enter interpreter path... (of Find...).
4. Vul het exacte pad naar de Python-executable in je venv in, bijvoorbeeld:
`/home/rudy/.venvs/fontys-ml/bin/python`
5. Open daarna een nieuwe terminal (Ctrl + Shift +  ) — VSCodium activeert de omgeving dan meestal automatisch (je ziet (fontys-ml)` voor de prompt staan).

Handmatig kan ook in de geïntegreerde terminal

Als je de omgeving alleen snel in je huidige terminalsessie wilt gebruiken:
1. Open de terminal in VSCodium met Ctrl +  ` (of via Terminal > New Terminal).
2. Voer het commando uit dat je al hebt, bijvoorbeeld:
`source /home/rudy/.venvs/fontys-ml/bin/activate`

## Stap 2: Python-script uitvoeren

Installeer eerst de benodigde libraries (requests en pandas):
pip install requests pandas librosa

Gebruik onderstaand script om voor 5 vogelsoorten elk 25 geluiden (kwaliteit A, zang) te downloaden in mappen per soort, inclusief een overzichtelijke dataset.csv:

## Python-script (Download + Preprocessing) download.py

```python
import os
import time
import requests
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

API_KEY = "Vul hier je EIGEN gratis Xeno Canto API in"
OUTPUT_DIR = "vogel_dataset_v1"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw_mp3")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "clips_wav")

AANTAL_PER_SOORT = 25
TARGET_SR = 22050       # Standaard sample rate voor veel audio-AI modellen
CLIP_DUUR = 4.0         # 4 seconden (valt mooi tussen 3-5 sec)
TARGET_SAMPLES = int(TARGET_SR * CLIP_DUUR)

# 5 klassen in Nederland voorkomende vogels
SOORTEN = [
    "Turdus merula",       # Merel
    "Erithacus rubecula",  # Roodborst
    "Parus major",         # Koolmees
    "Fringilla coelebs",   # Vink
    "Passer domesticus"    # Huismus
]

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
dataset_rijen = []

def extract_active_clip(y, sr, target_samples):
    """Pakt het luidste stuk van 4 sec, of vult aan met stilte als de clip korter is."""
    if len(y) <= target_samples:
        return np.pad(y, (0, target_samples - len(y)), mode="constant")
    
    # Bereken energie per venster van 0.5 sec om stiltes te vermijden
    hop = int(sr * 0.5)
    energies = [np.sum(y[i:i+target_samples]**2) for i in range(0, len(y) - target_samples, hop)]
    best_start = np.argmax(energies) * hop
    return y[best_start : best_start + target_samples]

def iter_opnames(query):
    """Levert opnames van opeenvolgende API-pagina’s."""
    url = "https://xeno-canto.org/api/3/recordings"
    pagina = 1
    while True:
        params = {"query": query, "key": API_KEY, "page": pagina}
        try:
            api_response = requests.get(url, params=params, timeout=30)
        except requests.RequestException:
            raise SystemExit("Xeno-canto is niet bereikbaar. Controleer je verbinding en probeer opnieuw.") from None
        if not api_response.ok:
            raise SystemExit(
                f"Xeno-canto geeft HTTP {api_response.status_code}. "
                "Controleer je API-sleutel en probeer later opnieuw."
            )
        try:
            response = api_response.json()
        except ValueError:
            raise SystemExit(
                "Xeno-canto geeft geen geldige JSON terug "
                f"(Content-Type: {api_response.headers.get('Content-Type', 'onbekend')}). "
                "Mogelijk geeft de server een HTML-foutpagina terug. Probeer later opnieuw."
            ) from None
        if not isinstance(response, dict) or not isinstance(response.get("recordings"), list):
            raise SystemExit("Onverwacht API-antwoord: de lijst 'recordings' ontbreekt of is ongeldig.")
        opnames = response["recordings"]
        if not opnames:
            return
        yield from opnames
        if pagina >= int(response["numPages"]):
            return
        pagina += 1


for klasse_id, soort in enumerate(SOORTEN):
    print(f"Verwerken: {soort}...")
    soort_naam = soort.replace(" ", "_")
    soort_wav_map = os.path.join(PROCESSED_DIR, soort_naam)
    os.makedirs(soort_wav_map, exist_ok=True)

    # Filter op soort, kwaliteit A, en zang of roep (huismussen tjilpen/roepen vaker dan ze 'zingen')
    genus, species = soort.split()
    query = f'gen:"{genus}" sp:"{species}" grp:birds q:A'
    aantal_gelukt = 0
    geziene_ids = set()

    for opname in iter_opnames(query):
        xc_id = opname["id"]
        if xc_id in geziene_ids:
            continue
        geziene_ids.add(xc_id)
        audio_url = opname["file"]
        raw_pad = os.path.join(RAW_DIR, f"XC{xc_id}.mp3")
        wav_pad = os.path.join(soort_wav_map, f"XC{xc_id}.wav")

        try:
            # 1. Download origineel indien nog niet aanwezig
            if not os.path.exists(raw_pad):
                audio_response = requests.get(audio_url, timeout=60)
                audio_response.raise_for_status()
                audio_data = audio_response.content
                with open(raw_pad, "wb") as f:
                    f.write(audio_data)
                time.sleep(0.5)

            # 2. Laad audio, resample naar 22050 Hz & converteer naar mono
            y, sr = librosa.load(raw_pad, sr=TARGET_SR, mono=True)

            # 3. Snijd luidste fragment van 4 seconden uit & normaliseer volume
            clip = extract_active_clip(y, sr, TARGET_SAMPLES)
            clip = librosa.util.normalize(clip)

            # 4. Sla op als uniforme .wav
            sf.write(wav_pad, clip, TARGET_SR)

            dataset_rijen.append({
                "xc_id": xc_id,
                "label_id": klasse_id,
                "klasse": soort,
                "bestandspad": wav_pad,
                "sample_rate": TARGET_SR,
                "duur_sec": CLIP_DUUR,
                "kwaliteit": opname.get("q"),
                "type_geluid": opname.get("type")
            })
            aantal_gelukt += 1
            if aantal_gelukt >= AANTAL_PER_SOORT:
                break
        except Exception as e:
            print(f"Fout bij XC{xc_id}: {e}")

    if aantal_gelukt < AANTAL_PER_SOORT:
        print(f"Let op: voor {soort} zijn slechts {aantal_gelukt} van de {AANTAL_PER_SOORT} clips verwerkt; er zijn geen verdere opnames beschikbaar.")

# Sla de dataset-index op als CSV
df = pd.DataFrame(dataset_rijen)
csv_pad = os.path.join(OUTPUT_DIR, "dataset_v1.csv")
df.to_csv(csv_pad, index=False)

print(f"Klaar! {len(df)} uniforme clips opgeslagen in {csv_pad}")
```

## Soorten toevoegen:
Bijvoorbeeld de Tjiftjaf ("Phylloscopus collybita") kun je eenvoudig toevoegen door de soortenlijst aan te passen. Omdat vogelopnames op Xeno-canto vaak stilte bevatten aan het begin of einde, kan men voor een V1-model het beste het luidste fragment van 4 seconden uit elke opname knippen en alles direct om zetten naar .wav op 22.050 Hz (mono).

## Belangrijke keuzes voor mijn V1-pipeline
## Omdat er niet alleen vogels met zang in de query zitten zoals de huismus:
• Huismus-query aangepast: De filter type:song is uit de zoekopdracht gehaald. Huismussen (Passer domesticus) staan op Xeno-canto meestal geregistreerd onder call (roep) in plaats van song (zang); zonder die filter krijgt men direct genoeg A-kwaliteit opnames.
• Luidste fragment (extract_active_clip): In plaats van blind de eerste 4 seconden te pakken (waar vaak alleen ruis of voetstappen van de opnemer op staan), zoekt deze functie het fragment met de hoogste geluidsenergie.
• Representatie voor het model: Met deze uniforme .wav-bestanden kan men in de trainingsloop direct Log-Mel Spectrogrammen genereren (bijv. librosa.feature.melspectrogram(y=clip, sr=22050, n_mels=128)), zodat elke audio-input exact dezelfde 2D-afmeting heeft voor een CNN of audio-classifier.

## Alternatief zonder programmeren (xcapi)

Wil je liever geen eigen script schrijven, dan kun je de officiële CLI-tool xcapi gebruiken. Deze downloadt direct audio en genereert automatisch een metadata.csv vanuit je terminal.
https://xeno-canto.org/article/323

## Omzetting naar Log-Mel Spectogrammen
Zoals in het "PvA Master piece vogelherkenning" is het handig om met een stukje code de .wav clips om te zetten naar Log-Mel Spectogrammen (als .npy -arrays of afbeeldingen) voor het gekozen framework of bibliotheek. Dit zou bijvoorbeeld PyTorch of Tensorflow als machine learning framework kunnen zijn. librosa: Dit is géén framework om zelflerende AI-modellen mee te bouwen. Het is een gereedschapskist specifiek voor geluid. Je gebruikt het om een audiobestand (zoals een .wav of .mp3) om te zetten naar getallen of visuele weergaven, zoals een Mel-spectrogram, zodat een AI-model het geluid kan "lezen".
(Mogelijk interessant: Voor PyTorch met audio, dan bestaat er blijkbaar ook torchaudio. Dat doet grotendeels hetzelfde als librosa, maar draait sneller op een videokaart (GPU)).

## Zie de readme *wav-spectogram-omzetten.md* voor de Python code om de .wav clips naar .npy om te zetten.
