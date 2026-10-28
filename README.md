# Demo Timestral

Confronto zero-shot tra Timestral-209M (Mistral) e Chronos-2 (Amazon) su due
benchmark industriali pubblici, con una pagina web per mostrare i risultati.

## Documenti

- [Sintesi per il manager](docs/sintesi-manager.md): licenze, fine-tuning, risultati e decisioni da prendere.
- [Cosa sanno fare i modelli](docs/modelli.md): input accettati, come lavorano i dati, output e limiti.

## Benchmark

| Dataset | Cosa si prevede | Riferimenti esterni | Licenza |
| --- | --- | --- | --- |
| **SDWPF / KDD Cup 2022** | potenza di 134 turbine eoliche a 48 ore, sui 142 casi di test ufficiali | vincitore della gara 44,92, baseline GRU 47,85 | CC BY 4.0 |
| **BDG2 / ASHRAE GEPIII** | consumo elettrico orario di 14 edifici industriali a 48 ore, 9 date nel 2017 | gara Kaggle con 3.614 team | CC BY-SA 4.0 |

## Installazione

```bash
uv sync
scripts/download.sh        # ~3 GB tra pesi dei modelli e dataset, esclusi da git
```

## Esecuzione

```bash
uv run python scripts/sdwpf_smoke.py --cases 142   # ~30 min su Mac M4
uv run python scripts/sdwpf_arima.py               # AutoARIMA su 20 casi, ~7 min
uv run python scripts/bdg2_smoke.py                # include AutoARIMA, ~8 min
uv run python scripts/export_frontend.py           # rigenera frontend/data.js in pochi secondi
open frontend/index.html
```

La pagina è statica e non ci sono dati inventati.
- **Metriche:** vengono dai CSV in `results/`.
- **Grafici:** usano le previsioni che i benchmark salvano in `results/forecasts/`, esclusa da git.
  - Per l'eolico si salvano solo i 5 casi mostrati nella pagina (`kddcup.SHOWCASE_CASES`).
  - Per BDG2 si salvano tutte le finestre.
- **Export:** legge solo la cache e i CSV, quindi richiede pochi secondi.

Per rigenerare solo la cache dell'eolico, senza toccare le metriche:

```bash
uv run python scripts/sdwpf_smoke.py --showcase-only   # ~1 min
uv run python scripts/sdwpf_arima.py --showcase-only   # ~3 min
```

## Struttura

| Percorso | Contenuto |
| --- | --- |
| [src/demo/models.py](src/demo/models.py) | interfaccia comune per Timestral, Chronos-2 e le baseline |
| [src/demo/kddcup.py](src/demo/kddcup.py) | caricamento dei casi KDD Cup e punteggio ufficiale della gara |
| [src/demo/bdg2.py](src/demo/bdg2.py) | caricamento BDG2, scelta degli edifici e delle finestre di test |
| [src/demo/metrics.py](src/demo/metrics.py) | MAE, WQL e copertura delle bande |
| [src/demo/cache.py](src/demo/cache.py) | cache delle previsioni tra benchmark ed export |
| [scripts/](scripts/) | benchmark, export per il frontend, download |
| [results/](results/) | metriche dei run completi (CSV) |
| [frontend/index.html](frontend/index.html) | pagina dei risultati |

## Avvertenze

- **Licenza Timestral:** i pesi sono sotto la "Modified MIT" di Mistral. Le aziende sopra i 20 M$ di fatturato mensile non hanno diritti d'uso senza una licenza commerciale.
- **Timestral su Mac:** gira su CPU, perché la sua normalizzazione usa float64 e la GPU Apple (MPS) non lo supporta. Chronos-2 usa MPS.
- **Dati già visti in addestramento:** `Salesforce/GiftEvalPretrain` contiene `kdd2022` e i siti BDG2 Bear, Fox, Panther e Rat.
  - BDG2: quei siti sono esclusi.
  - KDD Cup: i casi di test cadono tra maggio e ottobre 2021, dopo il periodo pubblicato per la gara. Però i dati completi sono pubblici dal 2024 e Mistral non dichiara i suoi dati di addestramento, quindi per Timestral non si può escludere.
- **BDG2, temperatura:** è quella effettivamente misurata, cioè una previsione meteo perfetta. In produzione il guadagno dalle covariate sarà più basso.
- **AutoARIMA:** un ARIMA stagionale con periodo di 24 ore costa 20-30 secondi per serie, perché il suo stato interno cresce col periodo. Per questo:
  - sui consumi si usa un ARIMA non stagionale, con i cicli giornaliero e settimanale passati come termini di Fourier (10 coppie ciascuno);
  - sull'eolico si stima sugli ultimi 3 giorni, e solo su 20 casi su 142.
  - Il numero di termini di Fourier è stato scelto su 20 finestre di test: con meno termini ARIMA non batte nemmeno "ripeti la settimana precedente".
