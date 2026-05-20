"""
Enhanced PDF Parser with OCR support for scanned PDFs
"""
import PyPDF2
import io

# Try to import OCR dependencies
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    import tempfile
    OCR_AVAILABLE = True
except ImportError:
    pass


async def parse_pdf_with_ocr(file_content: bytes, filename: str) -> dict:
    """
    Parse PDF with dual extraction methods:
    1. Text extraction (PyPDF2) - for text-based PDFs
    2. OCR (Tesseract) - fallback for scanned/image-based PDFs
    
    Args:
        file_content: Raw PDF bytes
        filename: Original filename
    
    Returns:
        dict with 'text', 'debug_info', or 'error'
    """
    
    if not file_content:
        return {"error": "File is empty."}
    
    # Method 1: Try standard text extraction
    text, extraction_info = extract_text_from_pdf(file_content)
    
    if text and len(text) > 50:
        return {
            "text": text,
            "debug_info": {
                "pages": extraction_info["pages"],
                "text_length": len(text),
                "extraction_method": "PyPDF2 (text-based PDF)"
            }
        }
    
    # Method 2: If extraction failed, try OCR
    if OCR_AVAILABLE:
        try:
            ocr_text, ocr_info = extract_text_with_ocr(file_content)
            if ocr_text and len(ocr_text) > 50:
                return {
                    "text": ocr_text,
                    "debug_info": {
                        "pages": ocr_info["pages"],
                        "text_length": len(ocr_text),
                        "extraction_method": "Tesseract OCR (scanned PDF)"
                    }
                }
        except Exception as ocr_err:
            pass
    
    # Both methods failed
    return {
        "error": "No readable text found in PDF. The file may be: (1) image-based/scanned without OCR support, (2) encrypted, or (3) using an unsupported format.",
        "debug_info": {
            "pages": extraction_info["pages"],
            "text_length": len(text),
            "extraction_method": "both_failed",
            "ocr_available": OCR_AVAILABLE
        }
    }


def extract_text_from_pdf(file_content: bytes) -> tuple:
    """Extract text from PDF using PyPDF2 (works for text-based PDFs)"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        
        if len(pdf_reader.pages) == 0:
            return "", {"pages": 0, "extraction_attempts": []}
        
        text = ""
        extraction_attempts = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                extracted = page.extract_text()
                if extracted and extracted.strip():
                    text += extracted + "\n"
                    extraction_attempts.append(f"Page {page_num + 1}: success ({len(extracted)} chars)")
                else:
                    extraction_attempts.append(f"Page {page_num + 1}: empty")
            except Exception as page_err:
                extraction_attempts.append(f"Page {page_num + 1}: error ({str(page_err)})")
                continue
        
        return text.strip(), {
            "pages": len(pdf_reader.pages),
            "extraction_attempts": extraction_attempts
        }
    except Exception as e:
        return "", {
            "pages": 0,
            "extraction_attempts": [f"PDF read error: {str(e)}"]
        }


def extract_text_with_ocr(file_content: bytes) -> tuple:
    """Extract text from scanned PDF using Tesseract OCR"""
    if not OCR_AVAILABLE:
        raise RuntimeError("Tesseract OCR not available")
    
    try:
        # Convert PDF to images
        images = convert_from_bytes(file_content, dpi=150)
        
        text = ""
        for page_num, image in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(image)
                if page_text and page_text.strip():
                    text += page_text + "\n"
            except Exception as page_err:
                print(f"OCR Page {page_num + 1} error: {page_err}")
                continue
        
        return text.strip(), {
            "pages": len(images),
            "ocr_method": "Tesseract"
        }
    except Exception as e:
        raise RuntimeError(f"OCR extraction failed: {str(e)}")
