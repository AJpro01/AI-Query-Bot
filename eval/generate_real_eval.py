import json
import chromadb
from pathlib import Path

def build_dataset():
    client = chromadb.PersistentClient("./chroma_db")
    collection = client.get_collection("book_chunks")

    # Fetch chunks
    results = collection.get(limit=60)
    
    eval_set = []
    
    for i in range(len(results['ids'])):
        chunk_id = results['ids'][i]
        text = results['documents'][i]
        metadata = results['metadatas'][i]
        
        # Skip TOC, front matter, or extremely short chunks
        if "...." in text or "Table of Contents" in text or len(text.strip()) < 150:
            continue
            
        # Clean text lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            continue
            
        # Extract meaningful context from the chunk
        chapter_section = metadata.get("section", metadata.get("chapter", "this topic"))
        if chapter_section in ["Unknown", "σ", ""]:
            chapter_section = lines[0][:50]

        # Construct a natural semantic question using chunk keywords
        snippet = " ".join(lines[:2])
        question = f"According to the text on {chapter_section}, what is explained about {snippet[:60]}?"

        eval_set.append({
            "id": f"q{len(eval_set) + 1}",
            "question": question,
            "ground_truth_chunk_id": chunk_id,
            "ground_truth_answer": text[:250].replace('\n', ' ').strip() + "..."
        })
        
        if len(eval_set) == 20:
            break

    # Save to eval/eval_set.json
    output_path = Path(__file__).parent / "eval_set.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)

    print(f"✅ Created {len(eval_set)} realistic Q&A entries in {output_path}")

if __name__ == "__main__":
    build_dataset()