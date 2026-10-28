# Timestral demo

Zero-shot comparison of Mistral's Timestral-209M against Chronos-2 and simple
baselines on two public industrial benchmarks with published results:

- **SDWPF / KDD Cup 2022**: 10-min power of 134 wind turbines, 48 h ahead,
  official final-phase test set (142 cases, 2,400+ competing teams).
  CC BY 4.0. Scored like the official evaluation (`src/demo/kddcup.py`).
- **BDG2 / ASHRAE GEPIII**: hourly electricity of industrial-type buildings
  (manufacturing, utility plants, warehouses, labs), 2017 test year. CC BY-SA 4.0.

## Setup

```bash
uv sync
scripts/download.sh        # ~3 GB: model weights + datasets, git-ignored
```

## Runs

```bash
uv run python scripts/sdwpf_smoke.py --cases 142
uv run python scripts/bdg2_smoke.py
```

Results land in `results/`.

## Caveats

- Timestral weights are under Mistral's Modified MIT license: no rights for
  companies above $20M monthly revenue without a commercial license.
- Timestral runs on CPU on Apple Silicon: its scaler uses float64, which MPS
  does not support. Chronos-2 uses MPS.
- `Salesforce/GiftEvalPretrain` contains `kdd2022` and the BDG2 sites Bear,
  Fox, Panther and Rat. Those BDG2 sites are excluded. The KDD Cup final test
  cases fall in May–Oct 2021, after the public 245-day training release, but
  the full 2020–2021 data has been public since 2024 and Timestral's training
  data is undisclosed, so SDWPF contamination cannot be ruled out for it.
