@echo off
if not exist results mkdir results
docker run --rm -v "%cd%\results:/results" nvcr.io/nvidia/tritonserver:23.12-py3-sdk perf_analyzer -m yolo -u host.docker.internal:8000 --concurrency-range 1:4:1 --measurement-interval 10000 --input-data=zero -f /results/benchmark.csv