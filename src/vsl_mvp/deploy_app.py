from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any
from uuid import uuid4
import base64
import unicodedata

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .config import FeatureConfigV2
from .infer import OnnxSignRecognizer
from .landmarks import LandmarkExtractor
from .landmarks_v2 import HolisticLandmarkExtractor


DEFAULT_MODEL_DIR = Path("runs/vsl_mvp30_v2_lite_transformer")
DEFAULT_LOG_DIR = Path("runs/deploy_tests")
DEFAULT_SAMPLE_VIDEO_DIR = Path("data/practice_videos")
VIDEO_EXTENSIONS = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "application/octet-stream": ".webm",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_slug(text: str, fallback: str = "unknown") -> str:
    value = re.sub(r"[^0-9A-Za-zÀ-ỹĐđ]+", "_", text.strip(), flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).replace("đ", "d").replace("Đ", "D")
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_marks.replace("_", " ").casefold().split())


def sample_label_from_path(path: Path) -> str:
    return path.stem.split("__", 1)[0].replace("_", " ")


def build_extractor(recognizer: OnnxSignRecognizer):
    schema_version = str(recognizer.config.get("schema_version", "v1_hands_pose"))
    if schema_version.startswith("v2"):
        config = FeatureConfigV2(sequence_length=int(recognizer.config["sequence_length"]))
        return HolisticLandmarkExtractor(config), schema_version, "MediaPipe Holistic V2"
    return LandmarkExtractor(), schema_version, "MediaPipe Hands+Pose V1"


class GoogleDriveUploader:
    def __init__(self, folder_id: str, credentials_path: str | Path) -> None:
        from google.oauth2 import service_account

        scopes = ["https://www.googleapis.com/auth/drive.file"]
        self.credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path),
            scopes=scopes,
        )
        self.folder_id = folder_id

    def upload(self, path: Path, *, mime_type: str | None = None) -> dict[str, str]:
        import requests
        from google.auth.transport.requests import Request

        metadata = {"name": path.name, "parents": [self.folder_id]}
        if not self.credentials.valid:
            self.credentials.refresh(Request())

        with path.open("rb") as f:
            files = {
                "metadata": ("metadata", json.dumps(metadata), "application/json; charset=UTF-8"),
                "file": (path.name, f, mime_type or "application/octet-stream"),
            }
            response = requests.post(
                "https://www.googleapis.com/upload/drive/v3/files",
                params={"uploadType": "multipart", "fields": "id,name,webViewLink", "supportsAllDrives": "true"},
                headers={"Authorization": f"Bearer {self.credentials.token}"},
                files=files,
                timeout=60,
            )
        response.raise_for_status()
        created = response.json()
        return {
            "id": str(created.get("id", "")),
            "name": str(created.get("name", path.name)),
            "webViewLink": str(created.get("webViewLink", "")),
        }


class DeployRecognizer:
    def __init__(self, model_dir: Path, log_dir: Path, sample_video_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_path = self._choose_model_path(model_dir)
        self.labels_path = model_dir / "labels.json"
        self.config_path = model_dir / "config.json"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.labels_path.exists() or not self.config_path.exists():
            raise FileNotFoundError(f"Missing labels/config in {model_dir}")

        self.recognizer = OnnxSignRecognizer(self.model_path, self.labels_path, self.config_path)
        self.extractor, self.schema_version, self.extractor_name = build_extractor(self.recognizer)
        self.labels = list(self.recognizer.labels)
        self.log_dir = log_dir
        self.video_dir = log_dir / "videos"
        self.log_path = log_dir / "attempts.jsonl"
        self.csv_log_path = log_dir / "attempts.csv"
        self.sample_video_dir = sample_video_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

        self.drive: GoogleDriveUploader | None = None
        self.webhook_url = os.getenv("LOG_WEBHOOK_URL", "").strip()
        self.webhook_secret = os.getenv("LOG_WEBHOOK_SECRET", "").strip()
        folder_id = os.getenv("GDRIVE_FOLDER_ID", "").strip()
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
        if credentials_json and not credentials_path:
            credentials_file = log_dir / "google-service-account.json"
            credentials_file.write_text(credentials_json, encoding="utf-8")
            credentials_path = str(credentials_file)
        if folder_id and credentials_path:
            self.drive = GoogleDriveUploader(folder_id, credentials_path)

        self.sample_videos = self._build_sample_video_map()

    @staticmethod
    def _choose_model_path(model_dir: Path) -> Path:
        int8_path = model_dir / "model.int8.onnx"
        return int8_path if int8_path.exists() else model_dir / "model.onnx"

    def close(self) -> None:
        self.extractor.close()

    def _build_sample_video_map(self) -> dict[str, Path]:
        if not self.sample_video_dir.exists():
            return {}
        videos = sorted(self.sample_video_dir.glob("*.mp4"))
        by_key = {text_key(sample_label_from_path(path)): path for path in videos}
        mapped = {}
        for label in self.labels:
            label_key = text_key(label)
            if label_key in by_key:
                mapped[label] = by_key[label_key]
                continue
            for sample_key, path in by_key.items():
                if label_key in sample_key or sample_key in label_key:
                    mapped[label] = path
                    break
        return mapped

    def save_upload(self, upload: UploadFile, member: str, target_label: str) -> tuple[Path, str]:
        content_type = upload.content_type or "application/octet-stream"
        extension = VIDEO_EXTENSIONS.get(content_type, Path(upload.filename or "").suffix or ".webm")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{safe_slug(member, 'member')}_{safe_slug(target_label, 'free_test')}_{uuid4().hex[:8]}{extension}"
        output_path = self.video_dir / name
        with output_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        return output_path, content_type

    def predict_video(self, path: Path) -> dict[str, Any]:
        with self.lock:
            extract_result = self.extractor.extract_video(path)
            if extract_result.status != "ok":
                quality = getattr(extract_result, "quality", {})
                return {
                    "status": extract_result.status,
                    "label": "",
                    "confidence": 0.0,
                    "top3": [],
                    "quality": quality,
                }
            prediction = self.recognizer.predict(extract_result.features)
            prediction["status"] = prediction.get("status", "ok")
            prediction["quality"] = getattr(extract_result, "quality", {})
            return prediction

    def append_log(self, payload: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        prediction = payload.get("prediction", {})
        quality = prediction.get("quality", {})
        top3 = prediction.get("top3", [])
        drive = payload.get("drive", {})
        drive_video = drive.get("video", {}) if isinstance(drive, dict) else {}
        row = {
            "timestamp": payload.get("timestamp", ""),
            "member": payload.get("member", ""),
            "target_label": payload.get("target_label", ""),
            "predicted_label": prediction.get("label", ""),
            "confidence": prediction.get("confidence", 0.0),
            "status": prediction.get("status", ""),
            "verified": payload.get("verified", False),
            "video_filename": payload.get("video_filename", ""),
            "drive_file_id": drive_video.get("id", ""),
            "drive_link": drive_video.get("webViewLink", ""),
            "top3_json": json.dumps(top3, ensure_ascii=False),
            "quality_json": json.dumps(quality, ensure_ascii=False),
        }
        write_header = not self.csv_log_path.exists()
        with self.csv_log_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def maybe_upload_to_drive(self, video_path: Path, content_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.drive is None:
            return {"enabled": False}

        try:
            video_info = self.drive.upload(video_path, mime_type=content_type)
        except Exception as exc:
            return {"enabled": True, "status": "upload_failed", "error": str(exc)}

        sidecar_path = Path(tempfile.gettempdir()) / f"{video_path.stem}.json"
        sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            log_info = self.drive.upload(sidecar_path, mime_type="application/json")
        except Exception as exc:
            return {
                "enabled": True,
                "status": "log_upload_failed",
                "video": video_info,
                "error": str(exc),
            }
        finally:
            sidecar_path.unlink(missing_ok=True)
        return {"enabled": True, "status": "ok", "video": video_info, "log": log_info}

    def maybe_send_to_webhook(self, video_path: Path, content_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.webhook_url:
            return {"enabled": False}

        import requests

        body = {
            "secret": self.webhook_secret,
            "filename": video_path.name,
            "content_type": content_type,
            "video_base64": base64.b64encode(video_path.read_bytes()).decode("ascii"),
            "attempt": payload,
        }
        try:
            response = requests.post(self.webhook_url, json=body, timeout=90)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return {"enabled": True, "status": "webhook_failed", "error": str(exc)}
        if data.get("ok") is False:
            return {
                "enabled": True,
                "status": "webhook_failed",
                "error": str(data.get("error", "Apps Script returned ok=false")),
                "response": data,
            }
        return {"enabled": True, "status": "ok", "response": data}


def create_app(
    model_dir: Path | None = None,
    log_dir: Path | None = None,
    sample_video_dir: Path | None = None,
) -> FastAPI:
    state = DeployRecognizer(
        model_dir=model_dir or Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)),
        log_dir=log_dir or Path(os.getenv("DEPLOY_LOG_DIR", DEFAULT_LOG_DIR)),
        sample_video_dir=sample_video_dir or Path(os.getenv("SAMPLE_VIDEO_DIR", DEFAULT_SAMPLE_VIDEO_DIR)),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.recognizer.close()

    app = FastAPI(title="VSL Team Test Deploy", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.recognizer = state

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML_PAGE

    @app.get("/api/labels")
    def labels() -> dict[str, Any]:
        recognizer: DeployRecognizer = app.state.recognizer
        sample_urls = [
            f"/api/sample/{idx}" if label in recognizer.sample_videos else None
            for idx, label in enumerate(recognizer.labels)
        ]
        sample_stream_urls = [
            f"/api/sample_stream/{idx}" if label in recognizer.sample_videos else None
            for idx, label in enumerate(recognizer.labels)
        ]
        storage_status = "none"
        if recognizer.drive is not None:
            storage_status = "google_drive_service_account"
        elif recognizer.webhook_url:
            storage_status = "apps_script_webhook"
        return {
            "model_dir": str(recognizer.model_dir),
            "model_path": str(recognizer.model_path),
            "num_labels": len(recognizer.labels),
            "labels": recognizer.labels,
            "sample_urls": sample_urls,
            "sample_stream_urls": sample_stream_urls,
            "schema_version": recognizer.schema_version,
            "extractor": recognizer.extractor_name,
            "google_drive_enabled": recognizer.drive is not None,
            "webhook_enabled": bool(recognizer.webhook_url),
            "storage_status": storage_status,
        }

    @app.get("/api/sample/{label_index}")
    def sample_video(label_index: int) -> FileResponse:
        recognizer: DeployRecognizer = app.state.recognizer
        if label_index < 0 or label_index >= len(recognizer.labels):
            raise HTTPException(status_code=404, detail="Sample not found.")
        label = recognizer.labels[label_index]
        path = recognizer.sample_videos.get(label)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="Sample not found.")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/sample_stream/{label_index}")
    def sample_stream(label_index: int) -> StreamingResponse:
        import time

        import cv2

        recognizer: DeployRecognizer = app.state.recognizer
        if label_index < 0 or label_index >= len(recognizer.labels):
            raise HTTPException(status_code=404, detail="Sample not found.")
        label = recognizer.labels[label_index]
        path = recognizer.sample_videos.get(label)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="Sample not found.")

        def frames():
            while True:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    break
                fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
                delay = min(max(1.0 / fps, 0.03), 0.12)
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                        if not ok:
                            continue
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buffer.tobytes()
                            + b"\r\n"
                        )
                        time.sleep(delay)
                finally:
                    cap.release()

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/attempt")
    def attempt(
        member: str = Form(...),
        target_label: str = Form(""),
        video: UploadFile = File(...),
    ) -> dict[str, Any]:
        recognizer: DeployRecognizer = app.state.recognizer
        if not video.content_type or not video.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File upload must be a video.")

        video_path, content_type = recognizer.save_upload(video, member, target_label)
        prediction = recognizer.predict_video(video_path)
        expected = target_label.strip()
        verified = bool(expected and prediction.get("label") == expected and prediction.get("status") == "ok")
        payload: dict[str, Any] = {
            "timestamp": utc_now(),
            "member": member.strip(),
            "target_label": expected,
            "video_path": str(video_path),
            "video_filename": video_path.name,
            "prediction": prediction,
            "verified": verified,
        }
        drive_info = recognizer.maybe_upload_to_drive(video_path, content_type, payload)
        payload["drive"] = drive_info
        webhook_info = recognizer.maybe_send_to_webhook(video_path, content_type, payload)
        payload["webhook"] = webhook_info
        recognizer.append_log(payload)
        return payload

    return app


HTML_PAGE = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VSL Team Test</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #15171a; }
    main { max-width: 980px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 16px; font-size: 28px; }
    section { background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; margin: 14px 0; }
    label { display: block; font-size: 13px; font-weight: 700; margin: 12px 0 6px; }
    input, select { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #c8ced8; border-radius: 6px; font-size: 15px; }
    video, img.sample-video { width: 100%; max-height: 420px; background: #111; border-radius: 8px; object-fit: contain; }
    #preview { transform: scaleX(-1); }
    button { border: 0; border-radius: 6px; padding: 10px 14px; font-size: 15px; font-weight: 700; cursor: pointer; margin: 10px 8px 0 0; }
    button.primary { background: #1f6feb; color: #fff; }
    button.secondary { background: #e7ebf2; color: #1b1d21; }
    button.danger { background: #cf222e; color: #fff; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    pre { white-space: pre-wrap; word-break: break-word; background: #111827; color: #e5e7eb; padding: 14px; border-radius: 8px; min-height: 80px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .video-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
    .video-title { font-size: 13px; font-weight: 700; margin: 0 0 8px; color: #4b5563; }
    @media (max-width: 720px) { .row, .video-grid { grid-template-columns: 1fr; } main { padding: 14px; } }
  </style>
</head>
<body>
  <main>
    <h1>VSL Team Test</h1>
    <section>
      <div class="row">
        <div>
          <label for="member">Thành viên</label>
          <input id="member" placeholder="Tên người test" />
        </div>
        <div>
          <label for="target">Kí hiệu cần test</label>
          <select id="target"></select>
        </div>
      </div>
    </section>
    <section>
      <div class="video-grid">
        <div>
          <p class="video-title">Video mau</p>
          <img id="sample" class="sample-video" alt="Video mau" />
        </div>
        <div>
          <p class="video-title">Camera test</p>
          <video id="preview" autoplay muted playsinline></video>
        </div>
      </div>
      <button id="start" class="primary">Bắt đầu quay</button>
      <button id="stop" class="danger" disabled>Dừng và gửi</button>
      <button id="retry" class="secondary" disabled>Quay lại</button>
    </section>
    <section>
      <pre id="result">Đang tải model...</pre>
    </section>
  </main>
  <script>
    const preview = document.getElementById('preview');
    const sample = document.getElementById('sample');
    const member = document.getElementById('member');
    const target = document.getElementById('target');
    const start = document.getElementById('start');
    const stopBtn = document.getElementById('stop');
    const retry = document.getElementById('retry');
    const result = document.getElementById('result');
    let stream, recorder, chunks = [], appInfo = null;

    function storageLabel(info) {
      if (info.google_drive_enabled) return 'Google Drive service account da bat';
      if (info.webhook_enabled) return 'Apps Script webhook da bat';
      return 'chua bat';
    }

    function updateSampleVideo() {
      if (!appInfo) return;
      const selectedIndex = target.selectedIndex - 1;
      const url = appInfo.sample_stream_urls && appInfo.sample_stream_urls[selectedIndex];
      if (url) {
        sample.src = `${url}?t=${Date.now()}`;
      } else {
        sample.removeAttribute('src');
      }
    }

    async function init() {
      const info = await fetch('/api/labels').then(r => r.json());
      appInfo = info;
      target.innerHTML = '<option value="">Không chọn nhãn</option>' +
        info.labels.map(label => `<option value="${label}">${label}</option>`).join('');
      result.textContent = `Model: ${info.num_labels} ki hieu\\nExtractor: ${info.extractor}\\nLuu tru: ${storageLabel(info)}`;
      target.addEventListener('change', updateSampleVideo);
      if (info.labels.length > 0) {
        target.selectedIndex = 1;
        updateSampleVideo();
      }
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      preview.srcObject = stream;
    }

    start.onclick = () => {
      chunks = [];
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';
      recorder = new MediaRecorder(stream, { mimeType });
      recorder.ondataavailable = event => { if (event.data.size > 0) chunks.push(event.data); };
      recorder.onstop = submit;
      recorder.start();
      result.textContent = 'Đang quay...';
      start.disabled = true;
      stopBtn.disabled = false;
      retry.disabled = true;
    };

    stopBtn.onclick = () => recorder && recorder.stop();
    retry.onclick = () => {
      chunks = [];
      result.textContent = 'Sẵn sàng quay lại.';
      retry.disabled = true;
      start.disabled = false;
    };

    async function submit() {
      stopBtn.disabled = true;
      result.textContent = 'Đang gửi video và chạy model...';
      const blob = new Blob(chunks, { type: 'video/webm' });
      const form = new FormData();
      form.append('member', member.value || 'unknown');
      form.append('target_label', target.value || '');
      form.append('video', blob, 'attempt.webm');
      const response = await fetch('/api/attempt', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) {
        result.textContent = JSON.stringify(data, null, 2);
      } else {
        const pred = data.prediction || {};
        const top3 = (pred.top3 || []).map(x => `${x.label}: ${(x.confidence * 100).toFixed(1)}%`).join('\\n');
        const webhook = data.webhook || {};
        const webhookResponse = webhook.response || {};
        const webhookLine = webhook.enabled
          ? `Webhook: ${webhook.status || 'unknown'}${webhook.error ? ` - ${webhook.error}` : ''}${webhookResponse.sheetName ? `\\nSheet: ${webhookResponse.sheetName}` : ''}${webhookResponse.fileUrl ? `\\nDrive file: ${webhookResponse.fileUrl}` : ''}`
          : 'Webhook: chua bat';
        result.textContent = `Dự đoán: ${pred.label || '(không nhận diện)'}\\nĐộ tin cậy: ${((pred.confidence || 0) * 100).toFixed(1)}%\\nTrạng thái: ${pred.status}\\nĐúng target: ${data.verified ? 'có' : 'không'}\\n${webhookLine}\\n\\nTop 3:\\n${top3}`;
      }
      retry.disabled = false;
      start.disabled = false;
    }

    init().catch(err => {
      result.textContent = 'Không khởi tạo được camera/app: ' + err.message;
    });
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VSL team testing web app.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", default=int(os.getenv("PORT", "8000")), type=int)
    parser.add_argument("--model-dir", default=Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)), type=Path)
    parser.add_argument("--log-dir", default=Path(os.getenv("DEPLOY_LOG_DIR", DEFAULT_LOG_DIR)), type=Path)
    parser.add_argument("--sample-video-dir", default=Path(os.getenv("SAMPLE_VIDEO_DIR", DEFAULT_SAMPLE_VIDEO_DIR)), type=Path)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(model_dir=args.model_dir, log_dir=args.log_dir, sample_video_dir=args.sample_video_dir),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
