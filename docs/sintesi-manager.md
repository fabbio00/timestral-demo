# Timestral: sintesi per il progetto

28 ottobre 2026

## In breve

- **Timestral** (Mistral, ottobre 2026) prevede serie temporali senza essere addestrato sui dati del cliente. Gira in locale, anche senza GPU.
- **Licenza:** è gratuito solo per le aziende sotto i **20 M$ di fatturato al mese** (circa 240 M$ l'anno), anche per le versioni fine-tunate. Sopra la soglia serve una licenza commerciale di Mistral, a sua discrezione e a prezzo non pubblico. Il cliente quasi certamente la supera.
- **Alternativa:** Chronos-2 (Amazon) ha licenza Apache 2.0, quindi uso commerciale libero e senza soglie.
- **Nei nostri test Timestral è il più preciso dei due**, con un margine contenuto.

## Risultati: previsione a 48 ore, senza addestramento

| Test pubblico | Timestral | Chronos-2 | Riferimento |
| --- | --- | --- | --- |
| Parco eolico, 134 turbine, 142 casi della KDD Cup 2022 (punteggio, più basso è meglio) | **46,98** | 49,25 | Vincitore della gara 44,92 · baseline ufficiale 47,85 |
| Consumi di 14 edifici industriali: riduzione dell'errore rispetto a "ripeti la settimana prima" | **−46%** | −41% | 0% |

Un metodo statistico classico (ARIMA) resta nettamente indietro:
- sui consumi riduce l'errore dell'11%, contro il 46% di Timestral;
- sull'eolico fa 57,5 contro 49,8, misurato sugli stessi 20 casi.

## Fine-tuning

- **Timestral:** solo tramite Mistral Forge, una piattaforma enterprise a pagamento.
- **Chronos-2:** il fine-tuning è incluso gratuitamente nella libreria open source.
- In entrambi i casi i guadagni attesi sono piccoli. Rende di più passare al modello informazioni note in anticipo, come il piano di produzione o il meteo.

## Rischi

- Timestral è in preview da due settimane. Il primo posto sui benchmark lo dichiara Mistral e nessuno l'ha ancora verificato.
- Sull'eolico non possiamo escludere che Timestral abbia già visto quei dati durante l'addestramento.
- Non sostituisce i modelli fisici del cliente: li integra.

## Decisioni richieste

1. Confermare il fatturato del cliente, per capire se la licenza di Timestral è un problema.
2. Far verificare la licenza a un legale prima di qualsiasi uso in produzione.
3. Recuperare l'esigenza precisa del cliente: quali dati ha e con che orizzonte deve prevedere.
