#!/usr/bin/env python3
"""Serve the local phone-boundary annotation interface."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from silent_speech_interpretability.data.textgrid import TextGridInterval, write_textgrid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "tools" / "phone_boundary_audit"
AUDIT_ROOT = PROJECT_ROOT / "artifacts" / "phone_boundary_audit"
METADATA_PATH = PROJECT_ROOT / "metadata" / "phone_boundary_audit_set.csv"
VALID_STATUSES = {"accepted", "corrected", "excluded", "unreviewed"}


def _load_data() -> dict:
    data = json.loads((AUDIT_ROOT / "data.json").read_text(encoding="utf-8"))
    for item in data["items"]:
        review_path = AUDIT_ROOT / "reviews" / f"{item['key']}.json"
        if review_path.exists():
            review = json.loads(review_path.read_text(encoding="utf-8"))
            item["phones"] = review["phones"]
            item["review"] = {"status": review["status"], "notes": review.get("notes", "")}
    return data


def _validate_review(payload: dict, source_item: dict) -> dict:
    status = str(payload.get("status", ""))
    if status not in VALID_STATUSES - {"unreviewed"}:
        raise ValueError("Choose accepted, corrected, or excluded before saving")
    phones = payload.get("phones")
    if not isinstance(phones, list) or len(phones) != len(source_item["phones"]):
        raise ValueError("The reviewed phone sequence must match the source sequence")
    normalized = []
    for current, source in zip(phones, source_item["phones"], strict=True):
        if current.get("arpabet") != source["arpabet"] or current.get("ipa") != source["ipa"]:
            raise ValueError("Phone labels cannot be changed in the boundary audit")
        start = float(current["start"])
        end = float(current["end"])
        if start < 0 or end > float(source_item["duration"]) + 1e-6 or end - start < 0.001:
            raise ValueError("Each phone must have a valid boundary and a duration of at least 1 ms")
        normalized.append({**source, "start": start, "end": end})
    for previous, current in zip(normalized, normalized[1:]):
        if current["start"] < previous["end"] - 1e-6:
            raise ValueError("Phone intervals cannot overlap")
    changed = any(
        abs(phone["start"] - phone["original_start"]) > 1e-6
        or abs(phone["end"] - phone["original_end"]) > 1e-6
        for phone in normalized
    )
    if status == "accepted" and changed:
        raise ValueError("Use corrected when boundaries have changed, or reset before accepting")
    if status == "corrected" and not changed:
        raise ValueError("Move at least one boundary before marking the item corrected")
    return {
        "key": source_item["key"],
        "user_id": source_item["user_id"],
        "group_name": source_item["group_name"],
        "status": status,
        "notes": str(payload.get("notes", "")).strip(),
        "phones": normalized,
    }


def _save_review(review: dict, source_item: dict) -> None:
    review_path = AUDIT_ROOT / "reviews" / f"{review['key']}.json"
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    reviewed_textgrid = AUDIT_ROOT / "reviewed_textgrids" / f"{review['key']}.TextGrid"
    words = [
        TextGridInterval(float(word["start"]), float(word["end"]), str(word["text"]))
        for word in source_item["words"]
    ]
    phones = [
        TextGridInterval(float(phone["start"]), float(phone["end"]), str(phone["ipa"]))
        for phone in review["phones"]
    ]
    write_textgrid(
        reviewed_textgrid,
        float(source_item["duration"]),
        {"words": words, "phones": phones},
    )
    metadata = pd.read_csv(METADATA_PATH).fillna("")
    match = (
        metadata.user_id.astype(str).eq(str(review["user_id"]))
        & metadata.group_name.astype(str).eq(str(review["group_name"]))
    )
    metadata.loc[match, "review_status"] = review["status"]
    metadata.loc[match, "review_notes"] = review["notes"]
    metadata.to_csv(METADATA_PATH, index=False)


class AuditHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        print(f"AUDIT_SERVER {self.address_string()} {format % args}", flush=True)

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/data":
            self._json(_load_data())
            return
        if path.startswith("/audit/"):
            relative = Path(path.removeprefix("/audit/"))
            target = AUDIT_ROOT / relative
            if relative.is_absolute() or ".." in relative.parts or not target.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            return self._file(target)
        if path == "/":
            target = WEB_ROOT / "index.html"
        else:
            target = (WEB_ROOT / path.lstrip("/")).resolve()
            if WEB_ROOT.resolve() not in target.parents or not target.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        return self._file(target)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            data = _load_data()
            source = next((item for item in data["items"] if item["key"] == payload.get("key")), None)
            if source is None:
                raise ValueError("Unknown audit item")
            base_data = json.loads((AUDIT_ROOT / "data.json").read_text(encoding="utf-8"))
            base_source = next(item for item in base_data["items"] if item["key"] == payload["key"])
            review = _validate_review(payload, base_source)
            _save_review(review, base_source)
            self._json({"ok": True, "review": {"status": review["status"], "notes": review["notes"]}})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not (AUDIT_ROOT / "data.json").exists():
        raise FileNotFoundError("Run scripts/52_prepare_phone_boundary_audit.py first")
    server = ThreadingHTTPServer((args.host, args.port), AuditHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Phone boundary audit available at {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
