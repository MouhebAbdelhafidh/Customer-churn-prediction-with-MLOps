from elasticsearch import Elasticsearch
import time

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Function to send logs to Elasticsearch
def log_to_elasticsearch(run_id, metric_name, value, step):
    """
    Log metrics to Elasticsearch.
    """
    doc = {
        "run_id": run_id,
        "metric": metric_name,
        "value": value,
        "step": step,
        "@timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    es.index(index="mlflow-metrics", document=doc)
    print(f"📊 {metric_name} logged to Elasticsearch: {value}")
