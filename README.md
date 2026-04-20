# GA Pricing Optimization Production Project

This project packages the notebook into a lightweight production-style service.

## What it does

- loads `retail_price.csv`
- auto-detects columns for `qty`, `unit_price`, `freight_price`
- runs a genetic algorithm over a single price multiplier
- saves:
  - `data/output/retail_price_suggested_prices.csv`
  - `data/output/optimization_summary.json`
- exposes a FastAPI service

## Project structure

```text
.
├── api/
├── src/
├── data/
│   ├── input/
│   │   └── retail_price.csv
│   └── output/
├── optimize_prices.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 1) Put your CSV here

```text
data/input/retail_price.csv
```

## 2) Install and run locally

```bash
pip install -r requirements.txt
python optimize_prices.py
```

## 3) Run API locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

## 4) Docker

```bash
docker compose up --build
```

## 5) Example REST request

```bash
curl -X POST "http://localhost:8000/optimize/path" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Recommended deployment strategy

For limited resources, keep it simple:

- use batch CLI as the main mode
- use FastAPI only if another app needs integration
- do not add Spark, Hadoop, Kafka, Kubernetes, Airflow, Oracle, ONNX, XGBoost, or Numba for this task
- optionally add MLflow later only if you start running many experiments with different elasticity assumptions
