import fitz
import sys
import shutil

print(f"Fitz version: {fitz.__doc__.splitlines()[0]}")
print(f"Tesseract in path: {shutil.which('tesseract')}")

files = [
    "Thesis_Data_Mining/Level_2_Meso/Google/Google.org 2025 Impact Report.pdf",
    "Thesis_Data_Mining/Level_3_Micro/Microsoft/Microsoft-AI-Diffusion-Report.pdf"
]

for pdf_path in files:
    print(f"--- File: {pdf_path} ---")
    try:
        doc = fitz.open(pdf_path)
        for i in range(min(2, len(doc))):
            page = doc[i]
            print(f"Page {i}:")
            try:
                tp = page.get_textpage_ocr(language='eng', dpi=300, full=True)
                content = page.get_text(page_hash=tp)
                print(f"  OCR Char Count: {len(content)}")
            except Exception as e:
                print(f"  OCR Error: {e}")
            
            std_content = page.get_text()
            print(f"  Standard Char Count: {len(std_content)}")
        doc.close()
    except Exception as e:
        print(f"Error opening file: {e}")
