#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排键模型本地 HTTP 服务。

用法：
    LAYOUT_CKPT=/path/to/ckpt_traj_full python3 serve_layout.py
    （ckpt 目录里要有 model.pth，可选 ckpt_state.json）

接口：
    GET  /v1/health                      -> {ready, feat, feature_set, device}
    POST /v1/layout  {"notes":[{t,type,dur,line_id}], "bpm":[{"t":sec,"bpm":120}], "level":0}
         -> {x:[...], src:["model"|"fallback"], n, n_model, n_fallback}
    x 是像素坐标，中心 0，右正，范围约 ±675。

环境变量：
    LAYOUT_CKPT              ckpt 目录（默认 ckpt_traj_full）
    LAYOUT_PORT              默认 8819
    LAYOUT_TOKEN             可选；设置后要求 Authorization: Bearer <token>
    LAYOUT_CORS_ORIGINS      逗号分隔的 origin，例如 http://127.0.0.1:4173
"""
import json, os, sys, tempfile, shutil, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import infer_traj
from traj_common import X_SCALE, X_CLIP, KEEP_TYPES, enforce_chord_gaps
from train_traj import TrajTransformer

CKPT_DIR = os.environ.get("LAYOUT_CKPT", "ckpt_traj_full")
PORT = int(os.environ.get("LAYOUT_PORT", "8819"))
TOKEN = os.environ.get("LAYOUT_TOKEN", "")
ORIGINS = [o.strip() for o in os.environ.get("LAYOUT_CORS_ORIGINS", "").split(",") if o.strip()]
LOCK = threading.Lock()

_ck = torch.load(os.path.join(CKPT_DIR, "model.pth"), map_location="cpu")
FEAT = int(_ck["model"]["proj.weight"].shape[1])
MODEL = TrajTransformer(feat=FEAT)
MODEL.load_state_dict(_ck["model"])
MODEL.eval()
STATE = {}
_state_path = os.path.join(CKPT_DIR, "ckpt_state.json")
if os.path.exists(_state_path):
    STATE = json.load(open(_state_path))
FEATURE_SET = STATE.get("feature_set") or ("base9" if FEAT >= 9 else "legacy8")
TARGET = STATE.get("target", "x")


def layout_notes(notes, bpm, level_norm=0.0):
    """入参音符列表 -> 每音 x（像素）。内部用临时目录喂给 infer_traj 的两遍推理。"""
    keep = []
    for index, note in enumerate(notes):
        note_type = int(note.get("type", 1) or 1)
        if note_type not in KEEP_TYPES:
            continue
        keep.append({"t": round(float(note["t"]), 6), "x": float(note.get("x", 0) or 0),
                     "type": note_type, "dur": float(note.get("dur", 0.0) or 0.0),
                     "line_id": int(note.get("line_id", 0) or 0), "index": index})
    if not keep:
        return {"x": [], "src": [], "n": 0, "n_model": 0, "n_fallback": 0}

    tmp = tempfile.mkdtemp(prefix="layout-")
    try:
        os.makedirs(os.path.join(tmp, "parsed"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "bpm"), exist_ok=True)
        json.dump({"fid": "live", "notes": [{k: n[k] for k in ("t", "x", "type", "dur", "line_id")} for n in keep]},
                  open(os.path.join(tmp, "parsed", "live.json"), "w"), ensure_ascii=False)
        segs = sorted([(float(seg["t"]), float(seg["bpm"])) for seg in (bpm or []) if float(seg.get("bpm", 0)) > 0])
        json.dump({"segs": segs}, open(os.path.join(tmp, "bpm", "live.json"), "w"))
        json.dump({}, open(os.path.join(tmp, "level_cache.json"), "w"))

        infer_traj.BASE = tmp
        with LOCK, torch.no_grad():
            pass1 = infer_traj.infer_notes("live", MODEL, {}, float(level_norm), feat_dim=FEAT,
                                           feature_set=FEATURE_SET, target=TARGET, audio=None)
            by_t = defaultdict(list)
            for note in pass1:
                by_t[round(note["t"], 3)].append(note["x"])
            chord_map = {t: float(np.clip(np.mean(xs) / X_SCALE, -X_CLIP, X_CLIP))
                         for t, xs in by_t.items() if len(xs) >= 2}
            out = infer_traj.infer_notes("live", MODEL, chord_map, float(level_norm), feat_dim=FEAT,
                                         feature_set=FEATURE_SET, target=TARGET, audio=None)
            out = enforce_chord_gaps(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 模型输出按 t 排序；按输入顺序回填（同 t 同 type 时按出现次序）
    bucket = defaultdict(list)
    for note in out:
        bucket[(round(note["t"], 4), int(note.get("type", 1)))].append(note)
    xs, srcs = [], []
    for note in keep:
        key = (round(note["t"], 4), note["type"])
        picks = bucket.get(key) or []
        picked = picks.pop(0) if picks else None
        xs.append(float(np.clip(picked["x"], -X_SCALE, X_SCALE)) if picked else 0.0)
        srcs.append(picked.get("src", "fallback") if picked else "fallback")
    return {"x": xs, "src": srcs, "n": len(xs), "n_model": sum(1 for s in srcs if s == "model"),
            "n_fallback": sum(1 for s in srcs if s != "model")}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def cors(self):
        origin = self.headers.get("Origin", "")
        allow = "*" if not ORIGINS else ("*" if origin in ORIGINS else "")
        if allow:
            self.send_header("Access-Control-Allow-Origin", allow)
            self.send_header("Access-Control-Allow-Headers", "Authorization,Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

    def json(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.cors()
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        return not TOKEN or self.headers.get("Authorization", "") == f"Bearer {TOKEN}"

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/health":
            return self.json(200, {"ready": True, "feat": FEAT, "feature_set": FEATURE_SET, "ckpt": CKPT_DIR})
        self.json(404, {"error": {"code": "not_found", "message": "unknown route"}})

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/layout":
            return self.json(404, {"error": {"code": "not_found", "message": "unknown route"}})
        if not self.authorized():
            return self.json(401, {"error": {"code": "unauthorized", "message": "bad token"}})
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as error:
            return self.json(400, {"error": {"code": "bad_json", "message": repr(error)[:120]}})
        notes = body.get("notes")
        if not isinstance(notes, list) or not notes:
            return self.json(400, {"error": {"code": "bad_notes", "message": "notes must be a non-empty list"}})
        try:
            result = layout_notes(notes, body.get("bpm") or [], float(body.get("level", 0) or 0))
        except Exception as error:
            return self.json(500, {"error": {"code": "layout_failed", "message": repr(error)[:200]}})
        return self.json(200, result)

    def log_message(self, fmt, *args):
        sys.stderr.write("[layout] " + fmt % args + "\n")


if __name__ == "__main__":
    print(f"排键模型服务 http://127.0.0.1:{PORT}  ckpt={CKPT_DIR} feat={FEAT} set={FEATURE_SET}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
