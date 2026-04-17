import requests
import json
payload={
  "inputs":[{
    "name":"RAW_INPUT",
    "shape":[1],
    "datatype":"BYTES",
    "data":["demo_image"]
  }]
}
r=requests.post("http://localhost:8000/v2/models/ensemble_yolo/infer",json=payload)
print(r.status_code)
print(r.text)