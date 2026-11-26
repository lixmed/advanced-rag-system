import re
from pypdf import PdfReader
def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)

    pages_text = []

    for page in reader.pages:
        txt = page.extract_text() or ""
        txt = txt.replace("\x00", " ")
        pages_text.append(txt)

    text = "\n\n".join(pages_text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

import fitz
import re

def read_pdf_text2(pdf_path: str):

    doc = fitz.open(pdf_path)

    pages_text = []

    for i, page in enumerate(doc):

        print(f"Reading page {i+1}/{len(doc)}")

        txt = page.get_text("text")

        txt = txt.replace("\x00", " ")

        txt = re.sub(r"[ \t]+", " ", txt)

        pages_text.append(f"\n[PAGE {i+1}]\n" + txt)

    text = "\n\n".join(pages_text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()