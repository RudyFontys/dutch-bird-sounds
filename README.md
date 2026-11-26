# dutch-bird-sounds

Dit is een Fontys schoolproject met AI en het herkennen van Nederlandstalige vogels.

![banner](https://github.com/RudyFontys/dutch-bird-sounds/blob/main/banner.png)

## Huidige status:

1. /docs/xeno-canto-import.md - Uitleg over het importeren uit Xeno Canto 
2. /docs/wav-spectogram-omzetten.md - uitleg over de conversie van wav naar spectogram
3. download.py - Python download code voor 125 samples van Xeno Canto
4. convert.py - Python conversie code voor omzetting van .wav audio naar spectogram
5. vogel_dataset_v1 - De volledige dataset bestaande uit 125 samples
6. [vogelgeluiden_baseline_v1.ipynb](vogelgeluiden_baseline_v1.ipynb) - Een eenvoudig CNN voor vijf vogelsoorten, met Nederlandse uitleg en uitgevoerde evaluatie

## Eerste notebook gebruiken

Open `vogelgeluiden_baseline_v1.ipynb` vanuit deze projectmap in Jupyter of VSCodium.
Selecteer de kernel **Python (fontys-ml)** (`/home/rudy/.venvs/fontys-ml/bin/python`)
en voer alle cellen van boven naar beneden uit. De benodigde pakketten zijn in deze
omgeving aanwezig. Voor een nieuwe Python-omgeving: `python -m pip install -r requirements-notebook.txt`;
selecteer daarna die omgeving als notebookkernel.

Het notebook leest de bestaande spectrogrammen en traint een klein CNN vanaf nul,
zonder augmentatie. De vaste verdeling richt zich op 75 trainings-, 25 validatie-
en 25 testopnamen. Exact gelijke spectrogrammen blijven in dezelfde set:
XC1167062 en XC1167063 bevatten dezelfde koolmeesopname, dus de 125 bestanden
bevatten 124 unieke spectrogrammen.

De uitvoer bevat leercurves, accuracy, precision/recall/F1 per soort, macro-F1,
een confusion matrix en beluisterbare foutanalyse. Modelgewichten, instellingen,
dataverdeling, CSV-tabellen en grafieken staan in
`resultaten/vogelgeluiden_baseline_v1/`. Opnieuw uitvoeren vervangt deze resultaten.

R. bouland
