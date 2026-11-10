"""Loopback-only web interface for the existing local literature agent."""
from __future__ import annotations

import hmac
import io
import json
import os
import re
import secrets
import shutil
import socket
import socketserver
import stat
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from importlib import resources
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote, urlsplit

from pypdf import PdfReader

from .index import IndexError_, build_index, load_index, render_page, save_index, verify_sources
from .models import DEFAULT_EMBEDDING, DEFAULT_VLM

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PAGES = 100
MAX_QUESTION_LENGTH = 2000


class WebError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def safe_filename(name: str) -> str:
    """Retain a readable basename without allowing paths or oversized names."""
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(c for c in name if c.isprintable()).strip().strip(".")
    if not name.lower().endswith(".pdf"):
        raise WebError(HTTPStatus.BAD_REQUEST, "请选择 PDF 文件。")
    stem = name[:-4].encode("utf-8")[:180].decode("utf-8", errors="ignore").strip(" .")
    return (stem or "paper") + ".pdf"


class WebService:
    """Persist one private library; run model work in a single background worker."""

    def __init__(self, data_dir: Path, model_name: str, embedding_model: str,
                 device: str = "auto", *, model_factory: Callable | None = None,
                 encoder_factory: Callable | None = None):
        self.data_dir = data_dir.expanduser().resolve()
        self.data_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.upload_dir = self.data_dir / "documents"
        self.upload_dir.mkdir(mode=0o700, exist_ok=True)
        self.index_path = self.data_dir / "index.json"
        self.model_name, self.embedding_model, self.device = model_name, embedding_model, device
        self._model_factory, self._encoder_factory = model_factory, encoder_factory
        self._model = self._encoder = None
        self._state_lock = threading.Lock()
        # PDFium and the model are shared between preview and background jobs.
        self._compute_lock = threading.RLock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-web")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active: str | None = None

    def close(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)

    def _get_model(self):
        if self._model is None:
            if self._model_factory is None:
                from .models import QwenVisionModel
                self._model_factory = QwenVisionModel
            self._model = self._model_factory(self.model_name, self.device)
        return self._model

    def _get_encoder(self):
        if self._encoder is None:
            if self._encoder_factory is None:
                from .models import DenseEncoder
                self._encoder_factory = DenseEncoder
            self._encoder = self._encoder_factory(self.embedding_model)
        return self._encoder

    def _index(self) -> dict:
        if self.index_path.exists():
            return load_index(self.index_path)
        return {"schema_version": 1, "documents": [], "chunks": [],
                "embedding_model": self.embedding_model}

    def documents(self) -> list[dict]:
        index = self._index()
        visual_ids = {c["document_id"] for c in index["chunks"] if c.get("kind") == "visual"}
        return [{"id": d["id"], "name": d["name"], "title": d["title"],
                 "pages": d["pages"], "visual_indexed": d["id"] in visual_ids}
                for d in index["documents"]]

    def document(self, document_id: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{16}", document_id):
            raise WebError(HTTPStatus.NOT_FOUND, "未找到这篇文献。")
        for doc in self._index()["documents"]:
            if doc["id"] == document_id:
                return doc
        raise WebError(HTTPStatus.NOT_FOUND, "未找到这篇文献。")

    def new_upload(self, filename: str) -> Path:
        filename = safe_filename(filename)
        parent = Path(tempfile.mkdtemp(prefix="paper-", dir=self.upload_dir))
        return parent / filename

    def busy(self) -> bool:
        with self._state_lock:
            return self._active is not None

    def _set_progress(self, job_id: str, progress: str) -> None:
        with self._state_lock:
            if job_id in self._jobs:
                self._jobs[job_id]["progress"] = progress

    def _submit(self, kind: str, action: Callable[[str], dict]) -> str:
        with self._state_lock:
            if self._active is not None:
                raise WebError(HTTPStatus.CONFLICT, "当前正在处理另一个任务，请稍候。")
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = {"id": job_id, "kind": kind, "state": "queued",
                                  "progress": "正在准备…", "result": None, "error": None}
            self._active = job_id
            # Keep completed job history bounded in long-running sessions.
            if len(self._jobs) > 60:
                for old_id in list(self._jobs):
                    if old_id != job_id and self._jobs[old_id]["state"] in {"done", "error"}:
                        del self._jobs[old_id]
                        if len(self._jobs) <= 40:
                            break

        def run():
            try:
                with self._state_lock:
                    self._jobs[job_id]["state"] = "running"
                with self._compute_lock:
                    result = action(job_id)
                with self._state_lock:
                    self._jobs[job_id].update(state="done", progress="完成", result=result)
            except Exception as exc:
                with self._state_lock:
                    self._jobs[job_id].update(state="error", progress="处理失败", error=str(exc))
            finally:
                with self._state_lock:
                    self._active = None

        try:
            self._worker.submit(run)
        except Exception:
            with self._state_lock:
                self._active = None
                self._jobs.pop(job_id, None)
            raise
        return job_id

    def job(self, job_id: str) -> dict:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise WebError(HTTPStatus.NOT_FOUND, "任务不存在，请重新操作。")
            return dict(job)

    def ingest(self, path: Path, describe_pages: bool) -> str:
        def action(job_id: str) -> dict:
            try:
                self._set_progress(job_id, "正在检查 PDF…")
                reader = PdfReader(str(path))
                if reader.is_encrypted and not reader.decrypt(""):
                    raise ValueError("暂不支持需要密码的 PDF。")
                pages = len(reader.pages)
                if not pages or pages > MAX_PAGES:
                    raise ValueError(f"PDF 页数需在 1 到 {MAX_PAGES} 页之间。")
                old = self._index()
                if old["documents"] and old.get("embedding_model") != self.embedding_model:
                    raise ValueError("当前文献库使用了不同的向量模型，请使用新的数据目录。")
                self._set_progress(job_id, "正在加载文本检索模型…")
                encoder = self._get_encoder()
                model = None
                if describe_pages:
                    self._set_progress(job_id, "正在加载视觉模型，首次运行可能较慢…")
                    model = self._get_model()

                def report(message: str) -> None:
                    page = re.search(r"page (\d+)/(\d+)", message)
                    if page:
                        self._set_progress(job_id, f"正在处理第 {page.group(1)} / {page.group(2)} 页…")
                    elif message.startswith("Computing embeddings"):
                        self._set_progress(job_id, "正在建立检索索引…")

                fresh = build_index([path], describe_page=model.describe if model else None,
                                    embed=encoder, progress=report)
                doc = fresh["documents"][0]
                # Re-uploading the same named PDF replaces its index entries, not any source file.
                old["documents"] = [d for d in old["documents"] if d["id"] != doc["id"]]
                old["chunks"] = [c for c in old["chunks"] if c["document_id"] != doc["id"]]
                old["documents"].append(doc)
                old["chunks"].extend(fresh["chunks"])
                old["embedding_model"] = self.embedding_model
                save_index(old, self.index_path)
                return {"document_id": doc["id"], "name": doc["name"], "pages": doc["pages"]}
            except Exception:
                # This directory was created exclusively for this upload.
                shutil.rmtree(path.parent, ignore_errors=True)
                raise
        return self._submit("ingest", action)

    def ask(self, question: str, document_id: str | None, mode: str) -> str:
        question = question.strip()
        if not question or len(question) > MAX_QUESTION_LENGTH:
            raise WebError(HTTPStatus.BAD_REQUEST, f"请输入 1 到 {MAX_QUESTION_LENGTH} 个字符的问题。")
        if not isinstance(mode, str) or mode not in {"auto", "text", "vision"}:
            raise WebError(HTTPStatus.BAD_REQUEST, "不支持的回答方式。")
        index = self._index()
        if document_id:
            self.document(document_id)
            index = {**index, "documents": [d for d in index["documents"] if d["id"] == document_id],
                     "chunks": [c for c in index["chunks"] if c["document_id"] == document_id]}
        if not index["documents"]:
            raise WebError(HTTPStatus.BAD_REQUEST, "请先上传一篇 PDF。")

        def action(job_id: str) -> dict:
            from .agent import ask
            self._set_progress(job_id, "正在加载模型，首次提问可能较慢…")
            model, encoder = self._get_model(), self._get_encoder()
            self._set_progress(job_id, "正在查找相关页面并生成回答…")
            result = ask(index, model, question, mode=mode, top_k=4, encoder=encoder,
                         max_visual_pages=2)
            # Trace excerpts can contain full paper text; the UI only needs the tool names.
            result["trace"] = [{k: t[k] for k in ("tool", "choice", "source", "page", "document") if k in t}
                               for t in result.get("trace", [])]
            return result
        return self._submit("ask", action)

    def page_png(self, document_id: str, page: int) -> bytes:
        doc = self.document(document_id)
        if not 1 <= page <= doc["pages"]:
            raise WebError(HTTPStatus.NOT_FOUND, "页码超出文献范围。")
        with self._compute_lock:
            verify_sources([doc])
            image = render_page(doc["path"], page, scale=1.5)
            buf = io.BytesIO()
            image.save(buf, format="PNG", optimize=True)
            return buf.getvalue()

    def pdf_bytes(self, document_id: str) -> tuple[bytes, str]:
        doc = self.document(document_id)
        verify_sources([doc])
        return Path(doc["path"]).read_bytes(), doc["name"]


class PrivateWebServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """A filesystem-protected socket can be forwarded over SSH to a browser."""
    daemon_threads = True

    def __init__(self, socket_path: Path, service: WebService):
        socket_path = socket_path.expanduser().resolve()
        socket_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if len(os.fsencode(socket_path)) >= 104:
            raise ValueError("Socket path is too long; use a shorter --socket path.")
        if socket_path.exists():
            info = socket_path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("Socket path exists and is not an owned Unix socket.")
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.connect(str(socket_path))
            except (ConnectionRefusedError, FileNotFoundError):
                socket_path.unlink(missing_ok=True)
            else:
                raise ValueError("The web interface is already running on this socket.")
        super().__init__(str(socket_path), WebHandler)
        os.chmod(socket_path, 0o600)
        self.socket_path = socket_path
        self._socket_inode = socket_path.lstat().st_ino
        self.service = service
        self.csrf_token = secrets.token_urlsafe(32)

    def server_close(self) -> None:
        super().server_close()
        try:
            if self.socket_path.lstat().st_ino == self._socket_inode:
                self.socket_path.unlink()
        except FileNotFoundError:
            pass


class WebHandler(BaseHTTPRequestHandler):
    server: PrivateWebServer

    def _send(self, status: int, content: bytes, content_type: str, **headers: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, value: Any) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _check_host(self) -> None:
        host = self.headers.get("Host", "")
        try:
            parsed = urlsplit("http://" + host)
            valid = parsed.hostname in {"127.0.0.1", "localhost"} and parsed.username is None
        except ValueError:
            valid = False
        if not valid:
            raise WebError(HTTPStatus.FORBIDDEN, "请通过本机地址访问。")

    def _check_token(self) -> None:
        token = self.headers.get("X-Local-Token", "")
        if not hmac.compare_digest(token, self.server.csrf_token):
            raise WebError(HTTPStatus.FORBIDDEN, "页面会话已过期，请刷新后重试。")

    def _length(self, maximum: int) -> int:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise WebError(HTTPStatus.LENGTH_REQUIRED, "无法读取请求长度。") from None
        if length <= 0:
            raise WebError(HTTPStatus.BAD_REQUEST, "请求内容为空。")
        if length > maximum:
            raise WebError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "文件或请求内容过大。")
        return length

    def do_GET(self) -> None:
        try:
            self._check_host()
            path = urlsplit(self.path).path
            if path in {"/", "/app.js", "/style.css"}:
                filename = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[path]
                content_type = {"index.html": "text/html", "app.js": "text/javascript",
                                "style.css": "text/css"}[filename] + "; charset=utf-8"
                asset = resources.files("literature_agent").joinpath("web_assets", filename).read_bytes()
                self._send(HTTPStatus.OK, asset, content_type,
                           Content_Security_Policy="default-src 'self'; img-src 'self' data:; "
                           "style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'")
            elif path == "/api/config":
                self._json(HTTPStatus.OK, {"token": self.server.csrf_token,
                                           "documents": self.server.service.documents(),
                                           "busy": self.server.service.busy(),
                                           "max_pdf_mb": MAX_PDF_BYTES // (1024 * 1024),
                                           "max_pages": MAX_PAGES})
            elif path == "/api/documents":
                self._json(HTTPStatus.OK, {"documents": self.server.service.documents()})
            elif match := re.fullmatch(r"/api/jobs/([a-f0-9]{32})", path):
                self._json(HTTPStatus.OK, self.server.service.job(match.group(1)))
            elif match := re.fullmatch(r"/api/documents/([a-f0-9]{16})/pages/(\d+)\.png", path):
                content = self.server.service.page_png(match.group(1), int(match.group(2)))
                self._send(HTTPStatus.OK, content, "image/png")
            elif match := re.fullmatch(r"/api/documents/([a-f0-9]{16})/file", path):
                content, name = self.server.service.pdf_bytes(match.group(1))
                self._send(HTTPStatus.OK, content, "application/pdf",
                           Content_Disposition="inline; filename*=UTF-8''" + quote(name, safe=""))
            else:
                raise WebError(HTTPStatus.NOT_FOUND, "页面不存在。")
        except WebError as exc:
            self._json(exc.status, {"error": exc.message})
        except (IndexError_, OSError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:
        upload: Path | None = None
        try:
            self._check_host()
            self._check_token()
            path = urlsplit(self.path).path
            if path == "/api/documents":
                if self.server.service.busy():
                    raise WebError(HTTPStatus.CONFLICT, "当前正在处理另一个任务，请稍候。")
                if self.headers.get_content_type() != "application/pdf":
                    raise WebError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "请选择 PDF 文件。")
                length = self._length(MAX_PDF_BYTES)
                query = parse_qs(urlsplit(self.path).query)
                name = query.get("filename", [""])[0]
                describe = query.get("describe", ["0"])[0] == "1"
                upload = self.server.service.new_upload(name)
                remaining = length
                with upload.open("wb") as stream:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise WebError(HTTPStatus.BAD_REQUEST, "上传中断，请重试。")
                        stream.write(chunk)
                        remaining -= len(chunk)
                with upload.open("rb") as stream:
                    if b"%PDF-" not in stream.read(1024):
                        raise WebError(HTTPStatus.BAD_REQUEST, "文件不是有效的 PDF。")
                job_id = self.server.service.ingest(upload, describe)
                upload = None  # The worker now owns the temporary upload.
                self._json(HTTPStatus.ACCEPTED, {"job_id": job_id})
            elif path == "/api/ask":
                if self.headers.get_content_type() != "application/json":
                    raise WebError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "请求格式不正确。")
                length = self._length(16 * 1024)
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    raise WebError(HTTPStatus.BAD_REQUEST, "请求格式不正确。") from None
                if not isinstance(body, dict) or not isinstance(body.get("question"), str):
                    raise WebError(HTTPStatus.BAD_REQUEST, "请输入问题。")
                doc_id = body.get("document_id")
                if doc_id is not None and not isinstance(doc_id, str):
                    raise WebError(HTTPStatus.BAD_REQUEST, "文献选择无效。")
                mode = body.get("mode", "auto")
                job_id = self.server.service.ask(body["question"], doc_id or None, mode)
                self._json(HTTPStatus.ACCEPTED, {"job_id": job_id})
            else:
                raise WebError(HTTPStatus.NOT_FOUND, "接口不存在。")
        except WebError as exc:
            if upload is not None:
                shutil.rmtree(upload.parent, ignore_errors=True)
                upload = None
            self._json(exc.status, {"error": exc.message})
        except (OSError, ValueError) as exc:
            if upload is not None:
                shutil.rmtree(upload.parent, ignore_errors=True)
                upload = None
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        finally:
            if upload is not None:
                shutil.rmtree(upload.parent, ignore_errors=True)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid recording questions, filenames, or query strings in access logs.
        return


def serve(socket_path: Path | None = None, data_dir: Path = Path("data/web"), model_name: str = DEFAULT_VLM,
          embedding_model: str = DEFAULT_EMBEDDING, device: str = "auto") -> None:
    service = WebService(data_dir, model_name, embedding_model, device)
    try:
        socket_path = socket_path or service.data_dir / "agent.sock"
        with PrivateWebServer(socket_path, service) as server:
            print(f"文献问答服务已启动，私有套接字：{server.socket_path}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        service.close()
