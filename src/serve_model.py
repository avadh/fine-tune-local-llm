from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = FastAPI()

# Load fine-tuned model
model_path = "models/fine_tuned_llm"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("mps")  # Apple Metal Backend
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map={"": device})

# Create a text-generation pipeline
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/v1/completions")
def generate_text(request: dict):
    """Generate a response from the fine-tuned model."""
    prompt = request.get("prompt", "")
    max_length = request.get("max_tokens", 100)

    if not prompt:
        return {"error": "No prompt provided."}

    response = qa_pipeline(prompt, max_length=max_length, do_sample=True)[0]["generated_text"]
    return {"response": response}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
