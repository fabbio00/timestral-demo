# Cosa sanno fare Timestral e Chronos-2

Sono **modelli di previsione già addestrati** (time series foundation model). Gli dai lo storico di una o più serie e restituiscono la previsione dei passi successivi con la relativa incertezza. Non vanno addestrati sui dati del cliente.

## Cosa accettano in input

| | Timestral-209M | Chronos-2 |
| --- | --- | --- |
| Serie da prevedere | una o più serie numeriche, anche correlate tra loro (multivariato) | idem |
| Storico massimo | 32.768 passi | 8.192 passi |
| Orizzonte per singola passata | fino a 1.024 passi, oltre si prosegue a blocchi | fino a 1.024 passi |
| Covariate solo passate (note fino a oggi) | sì | sì |
| Covariate note nel futuro (piano di produzione, calendario, previsioni meteo) | sì | sì |
| Covariate categoriche (turno, stato macchina, tipo di lotto) | sì | sì |
| Valori mancanti | ammessi (NaN) | ammessi (NaN) |
| Frequenza | qualsiasi, purché regolare: da secondi a mesi | idem |

**Cosa non accettano direttamente:** testi, immagini e log di eventi a intervalli irregolari. Vanno prima trasformati in serie regolari, per esempio "numero di allarmi per ora".

**Esempi nel contesto di un cantiere navale:**
- consumi energetici di reparti e impianti;
- sensori di macchine e gru;
- avanzamento della produzione o ore lavorate per settimana;
- consumi di carburante delle navi, con velocità e meteo come covariate.

## Come lavorano i dati

1. **Normalizzazione.** Ogni serie viene riportata su una scala comune, quindi kWh, °C o tonnellate per il modello sono la stessa cosa. Per questo funziona su dati mai visti.
2. **Patch.** Lo storico viene diviso in blocchi di punti consecutivi (32 per Timestral, 16 per Chronos-2), che il modello tratta un po' come le parole di una frase.
3. **Attenzione nel tempo e tra le serie.** Un transformer confronta i blocchi tra loro per riconoscere trend, stagionalità e cambi di regime. In alcuni livelli guarda anche le altre serie e le covariate fornite.
4. **Esperienza pregressa.** In addestramento hanno visto moltissime serie, in buona parte sintetiche (trend, stagionalità, rumore, dipendenze tra serie). Riconoscono gli schemi tipici e li applicano allo storico ricevuto: questo è lo "zero-shot".

## Cosa restituiscono

- **21 quantili per ogni passo futuro**, dall'1% al 99%.
- La **mediana** è la previsione "puntuale".
- La **fascia tra il 10% e il 90%** è l'intervallo in cui il valore reale dovrebbe cadere 8 volte su 10.
- La fascia di incertezza è utile anche come rilevatore di anomalie: un valore reale che esce dalla fascia è un segnale da controllare.

## Cosa non fanno

- Non spiegano il perché di una previsione e non conoscono la fisica del processo. Se una formula affidabile esiste già, rende di più come covariata che come alternativa.
- Non indovinano eventi senza traccia nello storico, come fermi impianto o festività. Bisogna passarli come covariate note nel futuro.
- La precisione cala con orizzonti lunghi. Nei benchmark pubblici rendono al meglio da qualche ora a qualche giorno.

## Tempi di calcolo

Misurati su un Mac M4, un modello alla volta. Previsione a 48 ore (288 passi) con 14 giorni di storico.

| Modello | Dove gira | 1 serie | 134 serie insieme |
| --- | --- | --- | --- |
| Timestral-209M | CPU | 0,15 s | 9,9 s |
| Chronos-2 | GPU del Mac | 0,03 s | 3,5 s |
| Chronos-2 | CPU | 0,08 s | 5,1 s |
| AutoARIMA | CPU, 10 processi | 0,05 s | 12,1 s |

- Di default Timestral calcola due previsioni, sulla serie e sulla serie ribaltata di segno, e ne fa la media: è più stabile, ma impiega il doppio del tempo.
- Su Mac Timestral non può usare la GPU, perché i suoi calcoli richiedono una precisione (float64) che la GPU Apple non supporta.
- Nessuno dei due modelli richiede hardware dedicato.

Fonti: [annuncio Timestral](https://mistral.ai/news/timestral/), [repository Timestral](https://github.com/mistralai/timestral), [Chronos-2 su Hugging Face](https://huggingface.co/amazon/chronos-2), file `config.json` dei due modelli.
