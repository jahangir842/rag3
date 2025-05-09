import os
import argparse
from pathlib import Path
import requests
from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_folder, max_chars=16000):
    combined_text = ""
    pdf_folder = Path(pdf_folder)

    for pdf_file in sorted(pdf_folder.glob("*.pdf")):
        try:
            reader = PdfReader(pdf_file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            combined_text += f"\n\n--- Document: {pdf_file.name} ---\n{text}\n"
            if len(combined_text) > max_chars:
                break
        except Exception as e:
            print(f"Failed to extract {pdf_file.name}: {e}")
    
    return combined_text.strip()[:max_chars]

def query_llama_server(prompt, server_url="http://localhost:8080/completion", max_tokens=256):
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
        "stop": ["###", "\n\n", "User:", "def ", "class "],
        "stream": False
    }

    response = requests.post(server_url, json=payload)

    print("\n=== Model Response ===\n")
    if response.status_code == 200:
        content = response.json()["content"]
        print(content)

        # Check if we hit the token limit (very basic)
        if len(content.strip().split()) >= max_tokens - 10:
            print("\n⚠️  Max token limit likely hit.")
        else:
            print("\n✅ Completed without hitting token limit.")
    else:
        print(f"[Error] HTTP {response.status_code}: {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query llama-server with PDF content as context.")
    parser.add_argument("--pdf_folder", required=True, help="Path to folder containing PDF files")
    parser.add_argument("--question", required=True, help="Your question for the model")
    args = parser.parse_args()

    context = extract_text_from_pdfs(args.pdf_folder)
    print("Context : ****************************** ", context)
    #final_prompt = f"""
    #You are analyzing multiple resumes. Each resume starts with: "--- Document: filename.pdf ---".

    #Your task is:
    #1. Go through **each resume independently**.
    #2. For the question below, check only the **explicit facts** mentioned in each document. Do not guess or infer.
    #3. If a match is found, output in the following format:

    #[filename.pdf] - [Candidate Name]: [Exact quote from resume that proves the answer]

    #4. If no relevant info is found, say:

    #[filename.pdf] - No relevant information found.

    #DO NOT hallucinate or make assumptions. Use only what's clearly written.

    ### Question:
    #{args.question}

    ### Resumes:
    #{context}

    #### Answer:
    #"""




    
    query_llama_server(final_prompt)
