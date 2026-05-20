import http.client
import pathlib
import uuid

pdf_path = pathlib.Path('test_resume.pdf')
content = pdf_path.read_bytes()

boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
body = b''
body += ('--' + boundary + '\r\n').encode('utf-8')
body += ('Content-Disposition: form-data; name="file"; filename="' + pdf_path.name + '"\r\n').encode('utf-8')
body += b'Content-Type: application/pdf\r\n\r\n'
body += content + b'\r\n'
body += ('--' + boundary + '--\r\n').encode('utf-8')

headers = {'Content-Type': 'multipart/form-data; boundary=' + boundary}
conn = http.client.HTTPConnection('127.0.0.1', 8000, timeout=20)
conn.request('POST', '/api/parse-pdf', body, headers)
res = conn.getresponse()
print(res.status, res.reason)
print(res.read().decode('utf-8'))
