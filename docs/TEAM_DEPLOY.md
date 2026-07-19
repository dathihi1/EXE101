# Team deploy for VSL testing

This app lets team members record a short sign clip in the browser, run the existing ONNX model, store a JSONL attempt log, and optionally upload each submitted video plus a JSON sidecar to Google Drive.

## Local run

```powershell
pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m vsl_mvp.deploy_app --host 0.0.0.0 --port 8000
```

If you run through Uvicorn directly, use the factory form:

```powershell
$env:PYTHONPATH="src"
uvicorn "vsl_mvp.deploy_app:create_app" --factory --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

Logs are written to:

```text
runs/deploy_tests/attempts.jsonl
runs/deploy_tests/attempts.csv
runs/deploy_tests/videos/
```

## Google Drive video storage

Use a Google Cloud service account:

1. Create a Google Cloud project and enable Google Drive API.
2. Create a service account and download its JSON key.
3. Create a Drive folder for test videos.
4. Share that folder with the service account email as Editor.
5. Set these environment variables before starting the server:

```powershell
$env:GDRIVE_FOLDER_ID="your_drive_folder_id"
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
$env:PYTHONPATH="src"
python -m vsl_mvp.deploy_app --host 0.0.0.0 --port 8000
```

When both variables are present, each submitted video is uploaded to Drive. A JSON sidecar for the same attempt is uploaded too, and the local JSONL log includes the Drive file IDs/links.

The CSV log is saved with UTF-8 BOM so Excel and Google Sheets can read Vietnamese text cleanly.

## Google Drive without service account keys

If Google Cloud blocks service account key creation with `iam.disableServiceAccountKeyCreation`, use Google Apps Script instead. This avoids service account keys entirely.

Create a Google Sheet for logs, then open:

```text
Extensions -> Apps Script
```

Paste this script:

```javascript
const DRIVE_FOLDER_ID = 'YOUR_DRIVE_FOLDER_ID';
const SHEET_NAME = 'attempts';
const SHARED_SECRET = 'CHANGE_THIS_RANDOM_SECRET';

function doPost(e) {
  const payload = JSON.parse(e.postData.contents);
  if (payload.secret !== SHARED_SECRET) {
    return jsonResponse({ ok: false, error: 'unauthorized' }, 403);
  }

  const attempt = payload.attempt || {};
  const prediction = attempt.prediction || {};
  const driveFolder = DriveApp.getFolderById(DRIVE_FOLDER_ID);
  const bytes = Utilities.base64Decode(payload.video_base64);
  const blob = Utilities.newBlob(bytes, payload.content_type || 'video/webm', payload.filename);
  const file = driveFolder.createFile(blob);

  const sheet = getSheet();
  if (sheet.getLastRow() === 0) {
    sheet.appendRow([
      'timestamp',
      'member',
      'target_label',
      'predicted_label',
      'confidence',
      'status',
      'verified',
      'video_filename',
      'drive_file_id',
      'drive_link',
      'top3_json',
      'quality_json'
    ]);
  }
  sheet.appendRow([
    attempt.timestamp || '',
    attempt.member || '',
    attempt.target_label || '',
    prediction.label || '',
    prediction.confidence || 0,
    prediction.status || '',
    attempt.verified || false,
    payload.filename || '',
    file.getId(),
    file.getUrl(),
    JSON.stringify(prediction.top3 || []),
    JSON.stringify(prediction.quality || {})
  ]);

  return jsonResponse({ ok: true, fileId: file.getId(), fileUrl: file.getUrl() });
}

function getSheet() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  return ss.getSheetByName(SHEET_NAME) || ss.insertSheet(SHEET_NAME);
}

function jsonResponse(data) {
  return ContentService
    .createTextOutput(JSON.stringify(data))
    .setMimeType(ContentService.MimeType.JSON);
}
```

Deploy it:

```text
Deploy -> New deployment -> Web app
Execute as: Me
Who has access: Anyone
```

Copy the Web app URL and add these Hugging Face Space secrets:

```text
LOG_WEBHOOK_URL=https://script.google.com/macros/s/.../exec
LOG_WEBHOOK_SECRET=CHANGE_THIS_RANDOM_SECRET
```

With this path, you do not need:

```text
GDRIVE_FOLDER_ID
GOOGLE_APPLICATION_CREDENTIALS
GOOGLE_APPLICATION_CREDENTIALS_JSON
```

Keep videos short because Apps Script receives the video as base64 JSON.

## Free deployment options

Recommended for this project:

1. Hugging Face Spaces with Docker
   - Good fit because this app needs Python, OpenCV, MediaPipe, and ONNX Runtime.
   - Free CPU Spaces are enough for team testing.
   - Runtime disk is not persistent by default, so keep Google Drive upload enabled.
   - Create a Space, choose Docker SDK, push this repo, and add these Space secrets:

```text
GDRIVE_FOLDER_ID=your_drive_folder_id
GOOGLE_APPLICATION_CREDENTIALS_JSON={...service account json...}
```

2. Render free web service
   - Also works for a private demo.
   - It may sleep when idle, so the first request can be slow.
   - Build command: `pip install -r requirements.txt`
   - Start command: `python -m vsl_mvp.deploy_app --host 0.0.0.0 --port $PORT`

3. Vercel/Netlify static frontend + backend elsewhere
   - Use this only if you split frontend and backend.
   - Do not put Google service account credentials in frontend code.

## Deploy notes

For a quick private team test, deploy on a VM or container service that supports Python, OpenCV, and MediaPipe. The browser camera requires HTTPS unless everyone tests on localhost, so put the app behind HTTPS when testing across devices.

Useful environment variables:

```text
MODEL_DIR=runs/vsl_mvp30_v2_lite_transformer
DEPLOY_LOG_DIR=runs/deploy_tests
PORT=8000
GDRIVE_FOLDER_ID=...
GOOGLE_APPLICATION_CREDENTIALS=...
GOOGLE_APPLICATION_CREDENTIALS_JSON=...
LOG_WEBHOOK_URL=...
LOG_WEBHOOK_SECRET=...
```
