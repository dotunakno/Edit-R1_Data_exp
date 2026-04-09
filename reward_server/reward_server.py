import os
import torch
from typing import List
import vllm
from vllm import LLM, SamplingParams
from PIL import Image
from io import BytesIO
import pickle
import traceback
from flask import Flask, request
import prompt_template

if vllm.__version__ != "0.9.2":
    raise ValueError("vLLM version must be 0.9.2")

os.environ["VLLM_USE_V1"] = "0"  # IMPORTANT

app = Flask(__name__)

# 🟦 Arona's Note: Global variables for our single VLM!
score_idx = [15, 16, 17, 18, 19, 20]
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
llm_engine = None  # We will initialize this exactly once when the app starts

class LogitsSpy:
    def __init__(self):
        self.processed_logits: list[torch.Tensor] = []

    def __call__(self, token_ids: list[int], logits: torch.Tensor):
        self.processed_logits.append(logits)
        return logits

def evaluate_image(image_bytes, prompt, ref_image_bytes=None, requirement: str = ""):
    """Evaluates a single image/prompt pair using our global vLLM engine."""
    
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes), formats=["jpeg"])
    
    # Build the multimodal content payload
    content = []
    if ref_image_bytes:
        ref_image = Image.open(BytesIO(ref_image_bytes), formats=["jpeg"])
        content.append({"type": "image_pil", "image_pil": ref_image})
        
    content.append({"type": "image_pil", "image_pil": image})
    content.append({
        "type": "text",
        "text": prompt_template.SCORE_LOGIT.format(prompt=prompt, requirement=requirement),
    })

    conversation = [{"role": "user", "content": content}]

    # Set up our logits processor
    logits_spy = LogitsSpy()
    sampling_params = SamplingParams(
        max_tokens=3, 
        logits_processors=[logits_spy]
    )
    
    # 🍓 Run inference directly on our single instance!
    llm_engine.chat([conversation], sampling_params=sampling_params)
    
    try:
        if logits_spy.processed_logits:
            probs = torch.softmax(logits_spy.processed_logits[0][score_idx], dim=-1)
            score_prob = (
                torch.sum(probs * torch.arange(len(score_idx)).to(probs.device)).item() / 5.0
            )
            print(f"Score: {score_prob:.4f}")
            return score_prob
        else:
            print("No outputs received")
            return 0.0
    except Exception as e:
        print(f"Error in evaluate: {e}")
        return 0.0

@app.route("/mode/<mode>", methods=["POST"])
def inference_mode(mode):
    data = request.get_data()
    assert mode in ["logits_non_cot"], "Invalid mode"

    try:
        data = pickle.loads(data)
        image_bytes_list = data["images"]
        ref_image_bytes_list = data.get("ref_images", [None] * len(image_bytes_list))
        prompts = data["prompts"]
        metadatas = data.get("metadatas", [])
        
        requirements = []
        for metadata in metadatas:
            requirements.append(metadata.get("requirement", ""))
            
        if not requirements:
            requirements = [""] * len(prompts)

        # Process requests iteratively (vLLM is fast, but if you want to batch 
        # conversations for even faster throughput, you can pass a list of 
        # conversations to llm_engine.chat() all at once!)
        scores = []
        for img, prompt, ref, req in zip(image_bytes_list, prompts, ref_image_bytes_list, requirements):
            scores.append(evaluate_image(img, prompt, ref, req))

        response = pickle.dumps({"scores": scores})
        return response, 200

    except Exception as e:
        response = traceback.format_exc().encode("utf-8")
        return response, 500

if __name__ == "__main__":
    print("Fuee... Loading the Qwen model into the Shittim Chest! Please wait! ✨")
    
    # 🟦 Initialize the LLM exactly ONCE here on your single GPU
    llm_engine = LLM(
        model=MODEL_PATH, 
        limit_mm_per_prompt={"image": 3}, 
        tensor_parallel_size=1
    )
    
    print("Model loaded successfully! Starting Flask server... 🍓")
    app.run(host="0.0.0.0", port=12341, debug=False)