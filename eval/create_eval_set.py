import json
import chromadb

def generate_template():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("book_chunks")
    
    # Fetch entries (request more so we can filter out front-matter)
    results = collection.get(limit=100)
    
    eval_entries = []
    count = 1
    
    for i in range(len(results['ids'])):
        chunk_id = results['ids'][i]
        text = results['documents'][i]
        metadata = results['metadatas'][i]
        
        # Skip title pages, TOC, and index pages containing dot leaders like "...."
        if "...." in text or "Table of Contents" in text or "Acknowledgments" in text:
            continue
            
        eval_entries.append({
            "id": f"q{count}",
            "question": f"TODO: Write a question about {metadata.get('chapter', 'this section')}",
            "ground_truth_chunk_id": chunk_id,
            "ground_truth_answer": text[:250].strip() + "..."
        })
        
        count += 1
        if count > 5:  # Stop after getting 5 solid body chunks
            break
    
    with open("eval/eval_set.json", "w", encoding="utf-8") as f:
        json.dump(eval_entries, f, indent=2)
        
    print("✅ Generated eval_set.json skipping front-matter/index pages!")

if __name__ == "__main__":
    generate_template()