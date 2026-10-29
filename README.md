# Demo Timestral

Confronto tra Timestral-209M (Mistral), Chronos-2 (Amazon) e AutoARIMA su due
benchmark industriali pubblici. Timestral e Chronos-2 lavorano zero-shot, cioè
senza essere addestrati su questi dati.

**Pagina dei risultati:** https://fabbio00.github.io/timestral-demo/

## Documenti

- [Sintesi per il progetto](docs/sintesi.md): licenze, fine-tuning, risultati e decisioni da prendere.
- [Cosa sanno fare i modelli](docs/modelli.md): input accettati, come lavorano i dati, output e limiti.

## I due benchmark

| | Parco eolico (SDWPF) | Edifici industriali (BDG2) |
| --- | --- | --- |
| Origine | KDD Cup 2022, oltre 2.400 team | dati della gara Kaggle ASHRAE GEPIII |
| Cosa si prevede | potenza di 134 turbine, 48 ore | consumo elettrico di 14 edifici, 48 ore |
| Input | 14 giorni di potenza, ogni 10 minuti | ~6 mesi di consumi orari, con o senza temperatura |
| Test | i 142 casi ufficiali della fase finale | 109 finestre da 9 date del 2017 |
| Punteggio | quello ufficiale della gara (MAE + RMSE in MW) | errore medio sul consumo reale, in % del consumo medio e in kWh; copertura delle bande |
| Confronto esterno | vincitore 44,92 · baseline GRU 47,85 | nessuno: la gara chiedeva un compito diverso |
| Licenza | CC BY 4.0 | CC BY-SA 4.0 |

La gara ASHRAE chiedeva di prevedere due anni interi a partire dal 2016. Qui
usiamo i suoi dati per prevedere le 48 ore successive, quindi la sua classifica
non è confrontabile.

## Come riprodurre

```bash
uv sync
scripts/download.sh                                  # ~3 GB tra modelli e dataset, esclusi da git
uv run python scripts/sdwpf_smoke.py --cases 142     # eolico, ~45 min su Mac M4
uv run python scripts/sdwpf_arima.py                 # AutoARIMA su 20 casi eolici, ~10 min
uv run python scripts/bdg2_smoke.py                  # edifici, tutti i modelli, ~10 min
uv run python scripts/timing.py                      # tempi a macchina scarica, ~3 min
uv run python scripts/export_frontend.py             # rigenera frontend/data.js in pochi secondi
uv run python scripts/sanity_checks.py               # 21 controlli su dati, modelli e metriche
```

- `scripts/progress.sh [secondi]` stampa a intervalli regolari a che punto sono i benchmark e quanto manca.
- `scripts/sdwpf_rescore.py` ricalcola i punteggi eolici dalle previsioni in cache, senza rifar girare i modelli.

`sanity_checks.py --official-metrics <path>` confronta il nostro punteggio
eolico con il codice ufficiale della gara (`metrics.py` in
[PaddleSpatial](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/paddlepaddle)).
L'ultima verifica ha dato una differenza massima di 3·10⁻⁷.

I benchmark salvano le previsioni in `results/forecasts/`, esclusa da git:
l'export e i ricalcoli del punteggio non devono rifar girare i modelli.

## Pagina dei risultati

[frontend/index.html](frontend/index.html) è una pagina statica: si apre anche
con un doppio clic. Legge `frontend/data.js`, generato dall'export: metriche dai
CSV in `results/` e serie dalle previsioni in cache, niente dati inventati.

A ogni push su `main` che tocca `frontend/`, il workflow
[pages.yml](.github/workflows/pages.yml) la pubblica su GitHub Pages.

## Struttura

| Percorso | Contenuto |
| --- | --- |
| [src/demo/models.py](src/demo/models.py) | interfaccia comune per Timestral, Chronos-2, AutoARIMA e le baseline |
| [src/demo/kddcup.py](src/demo/kddcup.py) | casi KDD Cup, pulizia dello storico, punteggio ufficiale |
| [src/demo/bdg2.py](src/demo/bdg2.py) | scelta degli edifici e delle finestre BDG2 |
| [src/demo/metrics.py](src/demo/metrics.py) | MAE, WQL e copertura delle bande |
| [src/demo/cache.py](src/demo/cache.py) | cache delle previsioni tra benchmark ed export |
| [scripts/](scripts/) | benchmark, tempi, controlli, export, download |
| [results/](results/) | metriche dei run (CSV) |
| [frontend/](frontend/) | pagina dei risultati e dati che legge |

## Scelte di metodo

**Modelli**
- Timestral e Chronos-2 girano con le impostazioni di default dei loro autori, senza messa a punto.
- Timestral su Mac usa la CPU: la sua normalizzazione richiede float64, che la GPU Apple (MPS) non supporta. Chronos-2 usa MPS e dà le stesse previsioni che su CPU.

**Eolico**
- I grafici mostrano 5 casi (presi tra i 20 di AutoARIMA, così ci sono tutti i modelli) e 5 turbine a intervalli regolari, per contenere il peso di `frontend/data.js`. Il punteggio conta tutti i 142 casi e le 134 turbine.
- Pulizia dello storico come indicano gli autori del dataset:
  - la potenza leggermente negativa (turbina ferma per vento debole) diventa 0;
  - sono sconosciuti, e arrivano ai modelli come mancanti, i punti con pale oltre 89°, quelli a zero con vento sopra 2,5 m/s, gli angoli impossibili e i dati mancanti.
- Il punteggio usa le regole del codice ufficiale: esclude i punti non validi e le turbine con previsione o valore reale tutto a zero.
- Quella esclusione premia chi prevede zero. La gara lo impediva scartando le consegne con previsioni costanti o con più del 10% di turbine a zero. Timestral e Chronos-2 passano sempre questi controlli, "ripeti l'ultimo valore" mai e AutoARIMA in 17 casi su 20: per questi ultimi il punteggio conta tutte le turbine.

**Edifici**
- Edifici industriali con almeno il 95% delle ore misurate, esclusi i 4 siti presenti nel corpus pubblico di addestramento `Salesforce/GiftEvalPretrain`.
- Escluse le finestre con il contatore spento per più di metà delle ore. Il filtro guarda anche le 48 ore da prevedere ed è uguale per tutti i modelli.
- La temperatura futura è quella misurata, cioè una previsione meteo perfetta: in produzione il guadagno sarà più basso.
- L'errore principale è la WAPE: MAE sulle 48 ore tenute fuori diviso il consumo medio di quelle ore (77 kWh/h). "Ripeti la settimana prima" è un modello di confronto: anche il suo errore è misurato sul reale.

**AutoARIMA**
- Un ARIMA stagionale con periodo di 24 ore costa 20-30 secondi per serie. Per questo si usa un ARIMA non stagionale con i cicli come termini di Fourier: 6 coppie giornaliere e 6 settimanali, scelte su finestre di validazione del 2016.
- Sull'eolico si stima sugli ultimi 3 giorni e gira su 20 casi su 142, per stare sotto i 10 minuti.

**Dati già visti in addestramento**
- Il corpus `GiftEvalPretrain` contiene anche `kdd2022`. I casi di test eolici (maggio-ottobre 2021) sono successivi ai dati pubblicati per la gara, ma il dataset completo è pubblico dal 2024.
- Chronos-2 dichiara di aver escluso questi benchmark. Mistral non dichiara i dati di addestramento di Timestral, quindi per Timestral non si può escludere.

## Dati e licenze

- **SDWPF:** Zhou et al., *Scientific Data* 2024, [CC BY 4.0](https://doi.org/10.6084/m9.figshare.24798654).
- **Building Data Genome 2:** Miller et al., *Scientific Data* 2020, [CC BY-SA 4.0](https://github.com/buds-lab/building-data-genome-project-2). Gli estratti in `frontend/data.js` sono ridistribuiti con la stessa licenza.
- **Timestral-209M:** pesi sotto la "Modified MIT" di Mistral: nessun diritto d'uso per le aziende sopra i 20 M$ di fatturato mensile senza licenza commerciale. I pesi non sono in questo repository.
- **Chronos-2:** Apache 2.0.
