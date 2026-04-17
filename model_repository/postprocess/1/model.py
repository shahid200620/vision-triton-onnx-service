import numpy as np
import triton_python_backend_utils as pb_utils
class TritonPythonModel:
    def execute(self, requests):
        responses=[]
        for request in requests:
            msg=np.array([b'{"status":"ok","detections":[]}'],dtype=object)
            out=pb_utils.Tensor("FINAL_OUTPUT",msg)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses