import onnxruntime as ort
import numpy as np
import torch

# Configuration (should match training config)
SRC_VOCAB_SIZE = 256 # Dummy value, will be determined by tokenizer
TGT_VOCAB_SIZE = 256 # Dummy value, will be determined by tokenizer
D_MODEL = 256
NUM_HEADS = 4
TINY_NUM_LAYERS = 2
D_FF = 512
DROPOUT = 0.1
MAX_LEN_SRC = 128
MAX_LEN_TGT = 64

def verify_onnx_model(onnx_path):
    try:
        # Load the ONNX model
        session = ort.InferenceSession(onnx_path)

        # Get input names
        input_names = [input.name for input in session.get_inputs()]

        # Create dummy input data matching the dynamic axes
        # Use random data within the vocab size
        dummy_src = np.random.randint(0, SRC_VOCAB_SIZE, size=(1, MAX_LEN_SRC)).astype(np.int64)
        dummy_tgt = np.random.randint(0, TGT_VOCAB_SIZE, size=(1, MAX_LEN_TGT - 1)).astype(np.int64)

        # Prepare inputs for ONNX Runtime
        inputs = {
            input_names[0]: dummy_src,
            input_names[1]: dummy_tgt
        }

        # Run inference
        outputs = session.run(None, inputs)

        print(f"ONNX model {onnx_path} loaded and inferred successfully.")
        print("Output shapes:")
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {output.shape}")

    except Exception as e:
        print(f"Error verifying ONNX model: {e}")

if __name__ == "__main__":
    onnx_model_path = "./tiny_medical_summarizer.onnx"
    verify_onnx_model(onnx_model_path)


