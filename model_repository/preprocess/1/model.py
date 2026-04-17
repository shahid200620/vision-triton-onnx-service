import numpy as np
import triton_python_backend_utils as pb_utils
class TritonPythonModel:
    def execute(self, requests):
        responses=[]
        for request in requests:
            arr=np.zeros((3,640,640),dtype=np.float32)
            out=pb_utils.Tensor("images",arr)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses