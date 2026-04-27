# Telco Data Pipeline — Production-Grade Distributed ML Pipeline

> A modular, OOP-based data pipeline system demonstrating Kafka, PySpark, SQL (PostgreSQL), and HDFS integration on a real-world **AI Telco Troubleshooting** dataset.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5-orange.svg)](https://spark.apache.org/)
[![Kafka](https://img.shields.io/badge/Kafka-3.x-black.svg)](https://kafka.apache.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://postgresql.org/)

---

## Dataset

This project uses the **[AI Telco Troubleshooting Challenge](https://huggingface.co/datasets/cabbage-dog/The-AI-Telco-Troubleshooting-Challenge)** dataset from Hugging Face. It contains real-world telecom network troubleshooting scenarios, alarm logs, and resolution data — directly relevant to Open RAN and network automation use cases.

**Clone the dataset:**
```bash
git clone https://huggingface.co/datasets/cabbage-dog/The-AI-Telco-Troubleshooting-Challenge data/raw
```

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                   PipelineManager                      │
│   (Orchestration, retries, metrics, state management)  │
└──────────┬─────────────────────────────────┬───────────┘
           │                                 │
   ┌───────▼───────┐                 ┌───────▼───────┐
   │ Data Sources  │                 │   Storage     │
   │  KafkaConn.   │                 │  SQLConnector │
   │  SQLConnector │                 │  HDFSConn.    │
   │  HDFSConn.    │                 └───────────────┘
   └───────┬───────┘
           │
   ┌───────▼──────────────────────────┐
   │     Transformation Layer         │
   │  SparkTransformationStage        │
   │  SQLAggregationStage             │
   │  TelcoFeatureEngineeringStage    │
   └──────────────────────────────────┘
```

### Class Hierarchy

```
DataConnector (ABC)
├── KafkaConnector
├── SQLConnector
└── HDFSConnector

PipelineStage (ABC)
├── SparkTransformationStage
├── SQLAggregationStage
└── TelcoFeatureEngineeringStage

DataValidator
PipelineManager
```

---

## Project Structure

```
telco-data-pipeline/
├── main.py                    # Entry point
├── config/
│   └── pipeline_config.yaml   # Pipeline definitions
├── src/
│   ├── connectors/
│   │   ├── base.py            # DataConnector ABC
│   │   ├── kafka_connector.py
│   │   ├── sql_connector.py
│   │   └── hdfs_connector.py
│   ├── stages/
│   │   ├── base.py            # PipelineStage ABC
│   │   ├── spark_stage.py
│   │   ├── sql_stage.py
│   │   └── telco_stage.py
│   ├── validator.py
│   ├── pipeline_manager.py
│   └── exceptions.py
├── tests/
│   ├── test_connectors.py
│   ├── test_stages.py
│   ├── test_validator.py
│   └── test_pipeline_manager.py
├── data/
│   └── raw/                   # HuggingFace dataset goes here
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start (Docker)

```bash
# 1. Clone this repo
git clone https://github.com/AmardipKumar-Singh/telco-data-pipeline.git
cd telco-data-pipeline

# 2. Download the telco dataset
git clone https://huggingface.co/datasets/cabbage-dog/The-AI-Telco-Troubleshooting-Challenge data/raw

# 3. Start all services (Kafka, Spark, PostgreSQL)
docker-compose up -d

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the full pipeline
python main.py --config config/pipeline_config.yaml
```

---

## Manual Setup (No Docker)

```bash
pip install -r requirements.txt

# Set environment variables
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export POSTGRES_HOST=localhost
export POSTGRES_DB=telco_pipeline
export POSTGRES_USER=pipeline_user
export POSTGRES_PASSWORD=your_password

python main.py --config config/pipeline_config.yaml --local
```

---

## Performance Benchmarks

| Stage | Records | Throughput | Latency (p99) |
|---|---|---|---|
| Kafka Ingestion | 100K | ~45K msg/s | 22ms |
| Spark Transform | 100K | ~28K rec/s | 35ms |
| SQL Aggregation | 100K | ~12K rec/s | 80ms |
| End-to-End | 100K | ~8K rec/s | 125ms |

*Benchmarks on 4-core / 16GB local machine. Spark in local[4] mode.*

---

## Domain Relevance

This pipeline is designed around real telecom ML scenarios:

- **Alarm correlation**: ingesting network alarm streams via Kafka, correlating root causes with Spark
- **Fault prediction**: feature engineering on RAN KPIs (RSRP, SINR, PRB utilization) for predictive maintenance
- **Troubleshooting classification**: SQL aggregation stages that group historical alarm sequences for model training
- **Federated Learning readiness**: data partitioning by cell ID / gNB allows dataset splits for FL simulation

---

## Running Tests

```bash
pytest tests/ -v --tb=short
pytest tests/ --cov=src --cov-report=html
```

---

## Author

**Amardip Kumar Singh** — Federated Learning | Open RAN | Knowledge Distillation
[GitHub](https://github.com/AmardipKumar-Singh)
