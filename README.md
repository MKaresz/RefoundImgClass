# RefoundImgClass
A machine learning model that automatically classifies refund items based on pictures of the items. It is running as a FastAPI service in a Docker image and can be triggered by calling the Client CLI API.

Training & Tracking: PyTorch + MLflow for experiment tracking and model registry. Artifacts stored remotely (S3) to support a containerized server app.  
Model Serving: Containerized FastAPI service that loads the MLflow‑registered model with alias @champion or, with a fallback to a model tagged "basemodel".
Batch Client: Asynchronous client that scans a date‑based folder, micro‑batches images, sends to server, writes JSONL ledger and daily Parquet, and runs simple analytics drift/imbalance checks. 
Orchestration: Timed PowerShell-registered Task Scheduler actions for inference and analytics on day's end. Client CLI flags provide ad‑hoc runs on chosen days, compatible with cron for nightly scheduling. 

# client.py --help
Usage: client.py [OPTIONS]

  batch-processing client that sends images to  FastAPI  inference server and 
  performs daily analytics.

  Tasks: Inference, Analytics, Reload-ML model     - gets server status     - 
  reads images from disk     - batches them     - sends them to the FastAPI   
  server     - logs predictions     - runs analytics     - calls model re-load
  on server

Options:
  --health          Get server status, info and version number.
  --inference       Inference on today's input images under: ./raw/YYYY-MM-DD/
  --analytics       Model analytics on yesterday's inferences:
                    ./predictions/YYYY-MM-DD.parquet
  --day [%Y-%m-%d]  Optional date for analytics (format: YYYY-MM-DD)
  --model_reload    Ask server to update model to latest champion or default  
                    model.
  --help            Show this message and exit.
