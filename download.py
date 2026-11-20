import os
import time
import requests
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

API_KEY = "Hier dus je EIGEN gratis API key van Xeno Canto invullen"
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