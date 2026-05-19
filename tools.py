from pathlib import Path
import re
import io
from collections import Counter

from config import CHUNK_OVERLAP, CHUNK_SIZE, ENABLE_OCR, OCR_DPI, OCR_LANG
from pypdf import PdfReader

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import fitz
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


def calculator(expression: str):
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"


def _normalize_text(text: str):
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str):
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2]


def _chunk_text(text: str, chunk_size: int, overlap: int):
    cleaned = _normalize_text(text)
    if not cleaned:
        return []

    chunks = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def _read_pdf_pages(file_path: str):
    reader = PdfReader(file_path)
    pages = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": index, "text": text.strip()})

    return pages


def _ocr_pdf_pages(file_path: str):
    if not ENABLE_OCR or fitz is None or pytesseract is None or Image is None:
        return []

    pages = []
    document = fitz.open(file_path)

    try:
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(dpi=OCR_DPI)
            image_bytes = pixmap.tobytes("png")
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang=OCR_LANG)
            pages.append({"page": index, "text": _normalize_text(text), "ocr": True})
    finally:
        document.close()

    return pages


def _iter_pdf_files(path: Path):
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            yield path
        return

    if path.is_dir():
        for file_path in sorted(path.rglob("*.pdf")):
            yield file_path


def load_knowledge_base(data_path: str):
    path = Path(data_path)

    if not path.exists():
        return None

    try:
        documents = []
        pdf_files = list(_iter_pdf_files(path))

        if not pdf_files and path.is_file():
            content = path.read_text(encoding="utf-8", errors="ignore")
            return {
                "type": "text",
                "path": str(path),
                "documents": [
                    {
                        "source": str(path),
                        "pages": [{"page": 1, "text": content}],
                        "chunks": [
                            {
                                "page": 1,
                                "chunk": 1,
                                "text": _normalize_text(content),
                            }
                        ],
                    }
                ],
            }

        for pdf_file in pdf_files:
            pages = _read_pdf_pages(str(pdf_file))
            ocr_pages = _ocr_pdf_pages(str(pdf_file))
            chunks = []

            for page_index, page in enumerate(pages, start=1):
                page_text = page.get("text", "")
                if not page_text and ocr_pages:
                    ocr_page = next((item for item in ocr_pages if item.get("page") == page_index), None)
                    if ocr_page and ocr_page.get("text"):
                        page_text = ocr_page["text"]
                        page["text"] = page_text
                        page["ocr"] = True

                for chunk_index, chunk in enumerate(
                    _chunk_text(page_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP),
                    start=1,
                ):
                    chunks.append(
                        {
                            "page": page["page"],
                            "chunk": chunk_index,
                            "text": chunk,
                            "ocr": page.get("ocr", False),
                        }
                    )

            documents.append(
                {
                    "source": str(pdf_file),
                    "pages": pages,
                    "chunks": chunks,
                    "ocr_enabled": ENABLE_OCR,
                    "ocr_available": fitz is not None and pytesseract is not None,
                }
            )

        return {
            "type": "pdf" if pdf_files else "text",
            "path": str(path),
            "documents": documents,
        }
    except Exception as exc:
        return f"Failed to load document: {exc}"


def knowledge_overview(knowledge_base):
    if knowledge_base is None:
        return "No document loaded."

    if isinstance(knowledge_base, str):
        return knowledge_base

    documents = knowledge_base.get("documents", [])
    total_pages = sum(len(document.get("pages", [])) for document in documents)
    extracted_pages = sum(
        1
        for document in documents
        for page in document.get("pages", [])
        if page.get("text")
    )
    total_chunks = sum(len(document.get("chunks", [])) for document in documents)
    ocr_documents = sum(1 for document in documents if document.get("ocr_enabled"))
    ocr_available = all(document.get("ocr_available") for document in documents) if documents else False
    first_chunks = [
        chunk.get("text", "")
        for document in documents
        for chunk in document.get("chunks", [])[:2]
        if chunk.get("text")
    ]
    preview = "\n\n".join(first_chunks) if first_chunks else "No text could be extracted from the document."

    return (
        f"Document type: {knowledge_base.get('type', 'unknown')}\n"
        f"Source: {knowledge_base.get('path', 'unknown')}\n"
        f"Files: {len(documents)}\n"
        f"Pages: {total_pages}\n"
        f"Pages with extracted text: {extracted_pages}\n"
        f"Chunks: {total_chunks}\n"
        f"OCR enabled: {'yes' if ocr_documents else 'no'}\n"
        f"OCR packages available: {'yes' if ocr_available else 'no'}\n"
        f"Preview:\n{preview}"
    )


def _page_text(page):
    return (page.get("text") or "").strip()


def search_knowledge_base(knowledge_base, user_query: str, limit: int = 5):
    retrieved_chunks = retrieve_relevant_chunks(knowledge_base, user_query, limit=limit)

    if isinstance(retrieved_chunks, str):
        return retrieved_chunks

    if not retrieved_chunks:
        return "No matching chunks found."

    return "\n".join(
        f"Match score {item['score']} | {item['source']} | page {item['page']} | chunk {item['chunk']} | OCR {item['ocr']}: {item['text'][:1200]}"
        for item in retrieved_chunks
    )


def retrieve_relevant_chunks(knowledge_base, user_query: str, limit: int = 5):
    if knowledge_base is None:
        return "No document loaded."

    if isinstance(knowledge_base, str):
        return knowledge_base

    query_terms = _tokenize(user_query)
    if not query_terms:
        return "No useful search terms found in the query."

    term_weights = Counter(query_terms)
    scored_chunks = []
    for document in knowledge_base.get("documents", []):
        source = document.get("source", "unknown")
        for chunk in document.get("chunks", []):
            text = _normalize_text(chunk.get("text", ""))
            tokens = _tokenize(text)
            if not tokens:
                continue

            token_counts = Counter(tokens)
            score = sum(token_counts[term] * weight for term, weight in term_weights.items())
            if score:
                scored_chunks.append(
                    (
                        score,
                        {
                            "source": source,
                            "page": chunk.get("page"),
                            "chunk": chunk.get("chunk"),
                            "text": chunk.get("text", ""),
                            "ocr": chunk.get("ocr", False),
                        },
                    )
                )

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    top_chunks = scored_chunks[:limit]

    return [
        {
            "score": score,
            **item,
        }
        for score, item in top_chunks
    ]


def page_summary(knowledge_base, page_number: int):
    if knowledge_base is None:
        return "No document loaded."

    if isinstance(knowledge_base, str):
        return knowledge_base

    for document in knowledge_base.get("documents", []):
        for page in document.get("pages", []):
            if page.get("page") == page_number:
                text = _page_text(page)
                if not text:
                    return f"Page {page_number} has no extractable text."
                return f"{document.get('source', 'unknown')} | Page {page_number}:\n{text[:2000]}"

    return f"Page {page_number} not found."


def build_document_context(knowledge_base, user_query: str, limit: int = 5):
    overview = knowledge_overview(knowledge_base)
    matches = search_knowledge_base(knowledge_base, user_query, limit=limit)

    if matches == "No matching chunks found." or matches == []:
        return overview

    if isinstance(matches, str):
        return f"{overview}\n\n{matches}"

    formatted_matches = "\n".join(
        f"Match score {item['score']} | {item['source']} | page {item['page']} | chunk {item['chunk']} | OCR {item['ocr']}: {item['text'][:1200]}"
        for item in matches
    )

    return f"{overview}\n\nRelevant pages for the current question:\n{formatted_matches}"


def get_research_brief(knowledge_base):
    if knowledge_base is None:
        return "No document loaded."

    if isinstance(knowledge_base, str):
        return knowledge_base

    return knowledge_overview(knowledge_base)


def format_citations(chunks):
    if not chunks or isinstance(chunks, str):
        return "No citations available."

    lines = []
    for item in chunks:
        lines.append(
            f"- {item['source']} | page {item['page']} | chunk {item['chunk']} | OCR {item['ocr']} | score {item['score']}"
        )
    return "\n".join(lines)


def get_tools():
    return {
        "calculator": calculator,
        "load_knowledge_base": load_knowledge_base,
        "knowledge_overview": knowledge_overview,
        "search_knowledge_base": search_knowledge_base,
        "page_summary": page_summary,
        "build_document_context": build_document_context,
        "get_research_brief": get_research_brief,
        "retrieve_relevant_chunks": retrieve_relevant_chunks,
        "format_citations": format_citations,
    }