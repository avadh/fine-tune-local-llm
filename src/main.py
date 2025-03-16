import os
import subprocess
import time
import requests


def run_extraction():
    """Runs the extract_text.py script to extract text from PDFs."""
    print("\n📂 Extracting text from PDFs...")
    result = subprocess.run(["python", "src/extract_text.py"], capture_output=True, text=True)
    print(result.stdout)


def run_fine_tuning():
    """Runs the fine_tune.py script to fine-tune the model."""
    print("\n🎯 Fine-tuning the model...")
    result = subprocess.run(["python", "src/fine_tune.py"], capture_output=True, text=True)
    print(result.stdout)


def start_llm_server():
    """Starts the LLM API server."""
    print("\n🚀 Starting LLM API server...")
    process = subprocess.Popen(["python", "src/serve_model.py"])
    time.sleep(5)  # Wait for the server to start
    return process


def test_model():
    """Tests the fine-tuned LLM API with a sample query."""
    url = "http://localhost:8000/v1/completions"
    payload = {"prompt": "What is prompt engineering?", "max_tokens": 100}

    print("\n📝 Testing the fine-tuned model...")
    try:
        response = requests.post(url, json=payload)
        response_json = response.json()
        print(f"\n💡 Model Response: {response_json['response']}")
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to the LLM server. Make sure it's running.")


def main():
    """Runs the full pipeline: extraction → fine-tuning → model serving → testing."""
    run_extraction()
    run_fine_tuning()

    llm_server_process = start_llm_server()

    test_model()

    input("\nPress Enter to stop the LLM server...")
    llm_server_process.terminate()


if __name__ == "__main__":
    main()
