import pdfplumber
import os
import json

# Define the directory where PDFs are stored
pdf_dir = "data/"
output_file = "data/fine_tuning_data.jsonl"

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n\n"
    return text.strip()

# Process all PDFs in the directory
data = []
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        print(f"Processing {pdf_path}...")
        text = extract_text_from_pdf(pdf_path)

        # Split text into paragraphs for fine-tuning format
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            if len(para.strip()) > 100:  # Avoid very short lines
                data.append({"instruction": "Explain this concept:", "response": para.strip()})

# Save extracted data in JSONL format
os.makedirs("data", exist_ok=True)
with open(output_file, "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Extracted text saved in {output_file}")
