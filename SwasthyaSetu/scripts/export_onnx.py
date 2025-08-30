
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.perspective_aware_transformer import TinyPerspectiveAwareMedicalSummarizer
from utils.tokenizer import MedicalTokenizer

# Paths (update as needed)
TOKENIZER_PATH = os.path.join("..", "model", "medical_tokenizer.json")
MODEL_PATH = os.path.join("..", "model", "demo_perspective_aware_model.pth")
ONNX_OUTPUT_PATH = os.path.join("..", "model", "perspective_aware_model.onnx")
MAX_LEN_SRC = 128
MAX_LEN_TGT = 64

def export_model_to_onnx(model_path, tokenizer_path, output_onnx_path):
    # Load tokenizer to get vocab size
    tokenizer = MedicalTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size_actual if hasattr(tokenizer, 'vocab_size_actual') else tokenizer.vocab_size

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model_config = checkpoint.get('model_config', {})
    model = TinyPerspectiveAwareMedicalSummarizer(
        vocab_size=vocab_size,
        **model_config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Dummy input for ONNX export
    dummy_src = torch.randint(0, vocab_size, (1, MAX_LEN_SRC))
    dummy_tgt = torch.randint(0, vocab_size, (1, MAX_LEN_TGT - 1))

    # Export the model
    torch.onnx.export(
        model,
        (dummy_src, dummy_tgt),
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["src", "tgt"],
        output_names=["vocab_dist", "copy_gate", "provenance_scores", "faithfulness_score", "attn_dist", "perspective_probs"],
        dynamic_axes={
            "src": {0: "batch_size", 1: "src_seq_len"},
            "tgt": {0: "batch_size", 1: "tgt_seq_len"},
            "vocab_dist": {0: "batch_size", 1: "tgt_seq_len"},
            "copy_gate": {0: "batch_size", 1: "tgt_seq_len"},
            "provenance_scores": {0: "batch_size", 1: "tgt_seq_len"},
            "faithfulness_score": {0: "batch_size"},
            "attn_dist": {0: "batch_size", 2: "tgt_seq_len", 3: "src_seq_len"},
            "perspective_probs": {0: "batch_size"}
        }
    )
    print(f"Model exported to {output_onnx_path}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model checkpoint not found at {MODEL_PATH}. Please train the model first.")
        exit(1)
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer file not found at {TOKENIZER_PATH}. Please train/build the tokenizer first.")
        exit(1)
    export_model_to_onnx(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        output_onnx_path=ONNX_OUTPUT_PATH
    )



