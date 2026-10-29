# Timestral: sintesi per il progetto

29 ottobre 2026 · [pagina dei risultati](https://fabbio00.github.io/timestral-demo/)

## In breve

- **Cos'è:** Timestral (Mistral, ottobre 2026) prevede serie temporali senza essere addestrato sui dati del cliente. Gira in locale, anche senza GPU.
- **Licenza, il punto critico:** è gratuito solo per le aziende sotto i **20 M$ di fatturato al mese** (circa 240 M$ l'anno), anche nelle versioni fine-tunate. Sopra la soglia serve una licenza commerciale di Mistral, a sua discrezione e a prezzo non pubblico.
- **Alternativa:** Chronos-2 (Amazon) ha licenza Apache 2.0, quindi uso commerciale libero.
- **Prestazioni:** nei nostri test Timestral è il più preciso. Chronos-2 è poco dietro e più veloce; il metodo statistico classico (ARIMA) resta nettamente indietro.

## Risultati: previsione a 48 ore, senza addestramento

| Test pubblico | Timestral | Chronos-2 | ARIMA | Riferimento |
| --- | --- | --- | --- | --- |
| Parco eolico, 142 casi della KDD Cup 2022 (punteggio della gara, più basso è meglio) | **47,5** | 49,9 | 57,9 ¹ | vincitore 44,9 · baseline ufficiale 47,9 |
| Consumi di 14 edifici industriali: errore in meno rispetto a "ripeti la settimana prima" | **−46%** | −41% | −2% | 0% |

¹ Misurato su 20 casi su 142, per contenere i tempi di calcolo. Sugli stessi casi Timestral fa 50,4.

- Il vincitore della gara aveva addestrato i suoi modelli su questi dati e usava anche il vento. Timestral vede solo la produzione passata.
- Il risultato sui consumi usa la temperatura come informazione aggiuntiva. Senza, Timestral riduce l'errore del 36%.

## Fine-tuning

- **Timestral:** solo tramite Mistral Forge, piattaforma enterprise a pagamento.
- **Chronos-2:** incluso gratuitamente nella libreria open source.
- In entrambi i casi i guadagni attesi sono piccoli. Rende di più dare al modello informazioni note in anticipo, come il piano di produzione o il meteo.

## Rischi

- Timestral è in preview da ottobre 2026. Il primo posto sui benchmark lo dichiara Mistral e nessuno l'ha ancora verificato in modo indipendente.
- Sull'eolico non possiamo escludere che Timestral abbia già visto quei dati in addestramento: Mistral non dichiara su cosa l'ha addestrato.
- Non sostituisce i modelli fisici del cliente: li integra.

## Decisioni richieste

1. Confermare il fatturato del cliente, per capire se la licenza di Timestral è un problema.
2. Far verificare la licenza a un legale prima di qualsiasi uso in produzione.
3. Recuperare l'esigenza precisa del cliente: quali dati ha e con che orizzonte deve prevedere.
