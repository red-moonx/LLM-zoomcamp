"""
Section-aware Parser & Chunker for Athletica Capstone RAG.
Reads full text files from Europe PMC (or PDFs/metadata) and PubMed JSON to produce structured chunks.
"""

import os
import json
import re
import pandas as pd
import pypdf
from typing import List, Dict, Any

FULLTEXT_DIR = "capstone-project/data/fulltext"
PDF_DIR = "capstone-project/data/pdfs"
JSON_PATH = "capstone-project/pubmed_recommended_papers.json"
OUTPUT_DIR = "capstone-project/data/processed"

def extract_pdf_text(pdf_path: str) -> str:
    """Extract full text from a PDF file using pypdf."""
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def split_text_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks by sections/paragraphs."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if len(current_chunk) + len(p_clean) + 2 <= chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + p_clean
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(p_clean) > chunk_size:
                for i in range(0, len(p_clean), chunk_size - overlap):
                    chunks.append(p_clean[i:i+chunk_size])
                current_chunk = ""
            else:
                current_chunk = p_clean
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    print(f"Processing {len(papers)} candidate papers for chunking...")
    all_chunks = []
    
    for paper in papers:
        pmid = paper["PMID"]
        pmc_id = paper.get("PMC Free Fulltext")
        title = paper.get("Title")
        authors = paper.get("Authors")
        category = paper.get("Category Name")
        pubdate = paper.get("Publication Date")
        
        txt_path = os.path.join(FULLTEXT_DIR, f"{pmid}_{pmc_id}.txt") if pmc_id and pmc_id.startswith("PMC") else None
        pdf_path = os.path.join(PDF_DIR, f"{pmid}_{pmc_id}.pdf") if pmc_id and pmc_id.startswith("PMC") else None
        
        full_text = ""
        source_type = "metadata"
        
        if txt_path and os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as tf:
                full_text = tf.read()
            if len(full_text.strip()) > 300:
                source_type = "fulltext_xml"
                
        if source_type == "metadata" and pdf_path and os.path.exists(pdf_path):
            pdf_text = extract_pdf_text(pdf_path)
            if len(pdf_text.strip()) > 500 and "Preparing to download" not in pdf_text:
                full_text = pdf_text
                source_type = "pdf"
                
        if source_type == "metadata":
            # Fallback to abstract & metadata
            abstract = paper.get("Abstract", "")
            full_text = f"Title: {title}\nAuthors: {authors}\nCategory: {category}\nPublication Date: {pubdate}\nJournal: {paper.get('Journal')}\nPMID: {pmid}\n\nAbstract:\n{abstract}"
            
        raw_chunks = split_text_into_chunks(full_text, chunk_size=1000, overlap=150)
        
        for idx, chunk_str in enumerate(raw_chunks, start=1):
            chunk_id = f"PMID_{pmid}_c{idx:02d}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "category_name": category,
                "publication_date": pubdate,
                "source_type": source_type,
                "chunk_index": idx,
                "content": chunk_str,
                "word_count": len(chunk_str.split()),
                "char_count": len(chunk_str)
            })
            
    print(f"\nGenerated total {len(all_chunks)} chunks across {len(papers)} papers.")
    
    # Save outputs
    json_out = os.path.join(OUTPUT_DIR, "chunks.json")
    csv_out = os.path.join(OUTPUT_DIR, "chunks.csv")
    
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    df = pd.DataFrame(all_chunks)
    df.to_csv(csv_out, index=False, encoding="utf-8")
    
    print(f"Exported chunks JSON to {json_out}")
    print(f"Exported chunks CSV to {csv_out}")

if __name__ == "__main__":
    main()
