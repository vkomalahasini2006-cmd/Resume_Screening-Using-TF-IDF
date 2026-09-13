import PyPDF2
try:
    with open("test_resume.pdf", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        print("Pages:", len(reader.pages))
        text = ""
        for p in reader.pages:
            text += p.extract_text()
        print("Extracted text length:", len(text))
        print("Text snippet:", text[:100])
except Exception as e:
    print("Error:", e)
