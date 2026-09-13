import requests

with open("test_resume.pdf", "rb") as f:
    r = requests.post("http://localhost:8000/api/parse-pdf", files={"file": ("test_resume.pdf", f, "application/pdf")})
    print(r.status_code)
    print(r.json())
