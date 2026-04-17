\# Vision Triton ONNX Service



Production-grade computer vision inference service using Triton Inference Server and ONNX.



\## Overview



This project exports a YOLOv8s model to ONNX, validates output parity with PyTorch, deploys the model using NVIDIA Triton Inference Server, enables dynamic batching, benchmarks performance, supports model versioning, and includes an ensemble pipeline with preprocessing and postprocessing stages.



\## Tech Stack



\* Python

\* PyTorch

\* Ultralytics YOLOv8

\* ONNX

\* ONNX Runtime

\* Triton Inference Server

\* Docker



\## Project Structure



\* scripts/

\* model\_repository/

\* results/

\* docker-compose.yml

\* README.md

\* .env.example



\## Features



\* Export YOLOv8s to ONNX

\* Validate ONNX output against PyTorch

\* Triton model serving

\* Dynamic batching

\* Benchmark report generation

\* Zero-downtime model versioning

\* Ensemble pipeline



\## Run Locally



1\. Create virtual environment

2\. Install dependencies

3\. Run export script

4\. Start Triton with Docker Compose

5\. Test endpoints



\## Main Commands



```cmd

python scripts\\export\_onnx.py

python scripts\\validate\_onnx.py

docker compose up

python scripts\\test\_ensemble.py

scripts\\test\_versioning.bat

```



\## Endpoints



\* \[http://localhost:8000/v2/health/ready](http://localhost:8000/v2/health/ready)

\* \[http://localhost:8000/v2/models/yolo](http://localhost:8000/v2/models/yolo)

\* \[http://localhost:8000/v2/models/ensemble\_yolo](http://localhost:8000/v2/models/ensemble\_yolo)



\## Benchmark



See `results/benchmark.csv`



\## Author



Shahid Mohammed



