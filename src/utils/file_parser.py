import io
from pypdf import PdfReader
import docx

def extract_text_from_file(filename: str, file_content: bytes) -> str:
    """Reads raw file bytes and extracts text based on the file extension."""
    text = ""
    filename_lower = filename.lower()
    
    if filename_lower.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(file_content))
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
                
    elif filename_lower.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_content))
        for para in doc.paragraphs:
            text += para.text + "\n"
            
    elif filename_lower.endswith(".txt"):
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            text = file_content.decode("latin-1")
        
    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
        
    return text

def chunk_text(text: str, max_words_per_chunk: int = 1000) -> list[str]:
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words_per_chunk):
        chunk = " ".join(words[i : i + max_words_per_chunk])
        chunks.append(chunk)
        
    return chunks