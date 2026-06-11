import io
from pypdf import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

_EXTENSION_KINDS = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}

# application/octet-stream is deliberately absent: it is the filler value most
# clients send when they don't know the type, so it carries no information.
_CONTENT_TYPE_KINDS = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}


def _kind_from_extension(filename: str) -> str | None:
    dot = filename.rfind(".")
    if dot == -1:
        return None
    return _EXTENSION_KINDS.get(filename[dot:].lower())


def _kind_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    # Strip parameters such as "; charset=utf-8".
    mime = content_type.split(";")[0].strip().lower()
    return _CONTENT_TYPE_KINDS.get(mime)


def _kind_from_magic_bytes(file_content: bytes) -> str | None:
    if file_content.startswith(b"%PDF-"):
        return "pdf"
    # DOCX is a ZIP container. Other OOXML formats share this signature, but as
    # a last resort it beats rejecting the file; a non-DOCX zip fails in parsing.
    if file_content.startswith(b"PK\x03\x04"):
        return "docx"
    return None


def detect_file_kind(filename: str, file_content: bytes, content_type: str | None = None) -> str | None:
    """Resolve a file to 'pdf' | 'docx' | 'txt', or None if unrecognized.

    The filename extension wins (it is also the stored identity of the file);
    the multipart part's declared Content-Type is the fallback for
    extension-less names, and magic bytes are the final, client-independent
    resort.
    """
    return (
        _kind_from_extension(filename)
        or _kind_from_content_type(content_type)
        or _kind_from_magic_bytes(file_content)
    )


def extract_text_from_file(filename: str, file_content: bytes, content_type: str | None = None) -> str:
    """Reads raw file bytes and extracts text based on the detected file kind."""
    kind = detect_file_kind(filename, file_content, content_type)
    text = ""

    if kind == "pdf":
        pdf = PdfReader(io.BytesIO(file_content))
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    elif kind == "docx":
        doc = docx.Document(io.BytesIO(file_content))
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif kind == "txt":
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            text = file_content.decode("latin-1")

    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")

    return text

def chunk_text(text: str, max_words_per_chunk: int = 1000) -> list[str]:
    chunk_size = max_words_per_chunk * 6  # ~6 chars per average word
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=max(100, chunk_size // 15),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text) or []
