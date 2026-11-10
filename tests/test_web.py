import http.client
import io
import json
import socket
import threading
import time
from urllib.parse import quote

import pytest
from reportlab.pdfgen import canvas

from literature_agent.web import PrivateWebServer, WebService, safe_filename


class FakeEncoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeModel:
    def __init__(self, *_args):
        pass

    def describe(self, _image):
        return "A chart compares Aspen and Cedar."

    def generate(self, prompt, images=None, max_new_tokens=0):
        if "Return ONLY JSON" in prompt:
            return '{"needs_vision": false}'
        return '{"answer": "120 research documents", "citations": ["E1"]}'


@pytest.fixture
def web_server(tmp_path):
    service = WebService(tmp_path, "fake-vlm", "fake-encoder", model_factory=FakeModel,
                         encoder_factory=FakeEncoder)
    server = PrivateWebServer(tmp_path / "agent.sock", service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    service.close()
    thread.join(timeout=2)


def request(server, method, path, body=None, headers=None):
    class UnixConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(str(server.socket_path))

    conn = UnixConnection("localhost", timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def get_json(server, path):
    status, _, body = request(server, "GET", path)
    return status, json.loads(body)


def wait_for_job(server, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, job = get_json(server, f"/api/jobs/{job_id}")
        if job["state"] in {"done", "error"}:
            return job
        time.sleep(0.02)
    raise AssertionError("web job did not finish")


def sample_pdf():
    out = io.BytesIO()
    document = canvas.Canvas(out)
    document.drawString(40, 750, "The benchmark contains 120 research documents.")
    document.showPage()
    document.drawString(40, 750, "Figure 1 compares Aspen and Cedar.")
    document.save()
    return out.getvalue()


def test_web_upload_answer_and_citation_preview(web_server):
    status, config = get_json(web_server, "/api/config")
    assert status == 200
    token = config["token"]
    body = sample_pdf()
    url = "/api/documents?filename=" + quote("示例.pdf") + "&describe=1"
    status, _, payload = request(web_server, "POST", url, body,
                                  {"Content-Type": "application/pdf", "X-Local-Token": token})
    assert status == 202
    uploaded = wait_for_job(web_server, json.loads(payload)["job_id"])
    assert uploaded["state"] == "done", uploaded
    doc_id = uploaded["result"]["document_id"]

    _, library = get_json(web_server, "/api/documents")
    assert len(library["documents"]) == 1
    assert library["documents"][0]["name"] == "示例.pdf"
    assert library["documents"][0]["visual_indexed"] is True
    assert "path" not in library["documents"][0]

    question = json.dumps({"question": "How many research documents?", "document_id": doc_id,
                           "mode": "text"}).encode()
    status, _, payload = request(web_server, "POST", "/api/ask", question,
                                  {"Content-Type": "application/json", "X-Local-Token": token})
    assert status == 202
    answered = wait_for_job(web_server, json.loads(payload)["job_id"])
    assert answered["state"] == "done", answered
    assert answered["result"]["answer"] == "120 research documents [E1]"
    assert answered["result"]["citations"][0]["document_id"] == doc_id
    assert answered["result"]["citations"][0]["page"] == 1
    assert "excerpt" not in json.dumps(answered["result"]["trace"])

    status, headers, image = request(web_server, "GET", f"/api/documents/{doc_id}/pages/1.png")
    assert status == 200 and headers["Content-Type"] == "image/png" and image.startswith(b"\x89PNG")
    status, headers, original = request(web_server, "GET", f"/api/documents/{doc_id}/file")
    assert status == 200 and headers["Content-Type"] == "application/pdf" and original == body
    assert request(web_server, "GET", f"/api/documents/{doc_id}/pages/3.png")[0] == 404


def test_web_rejects_untrusted_requests_and_invalid_files(web_server):
    assert request(web_server, "GET", "/api/config", headers={"Host": "example.com"})[0] == 403
    assert request(web_server, "POST", "/api/ask", b"{}", {"Content-Type": "application/json"})[0] == 403
    _, config = get_json(web_server, "/api/config")
    token = config["token"]
    status, _, _ = request(web_server, "POST", "/api/documents?filename=bad.pdf", b"not a pdf",
                           {"Content-Type": "application/pdf", "X-Local-Token": token})
    assert status == 400
    assert list(web_server.service.upload_dir.iterdir()) == []
    assert safe_filename("../../论文.pdf") == "论文.pdf"
    assert safe_filename("C:\\folder\\paper.pdf") == "paper.pdf"
