# QKD-SAT

## Run Locally

Install dependencies

```bash
  pip install -U requirements.txt
```

Run simulations

```bash
  python sim.py --num-nodes <N> [--new-graph] --num-runs <NR> --bin-size <BS> --vary both 
```

The output is saved in the [results](./results) directory and graphs are stored in the [graphs](./graphs) directory.

Render the results using

```bash
python chart.py
```
