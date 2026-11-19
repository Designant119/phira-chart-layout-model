#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, sys, os

chart_dir, out_mp4 = sys.argv[1], sys.argv[2]
player = sys.argv[3] if len(sys.argv) > 3 else "Doubao"
name = sys.argv[4] if len(sys.argv) > 4 else "chart (model)"
w, h = 480, 270
if len(sys.argv) > 5:
    w, h = [int(v) for v in sys.argv[5].split('x')]

info = {
    "id": 15831, "uploader": 15831,
    "name": name,
    "difficulty": 15.3,
    "level": "IN Lv.15",
    "charter": "shabu5553",
    "composer": "塞壬唱片-MSR",
    "illustrator": "UK",
    "chart": "generated_chart.json",
    "music": "38153333.mp3",
    "illustration": "38153333.jpg",
    "previewStart": 0.0,
    "aspectRatio": 1.7777778,
    "backgroundDim": 0.6,
    "lineLength": 6.0,
    "offset": 0.0,
}
config = {
    "resolution": [w, h],
    "fps": 15,
    "sampleCount": 1,
    "challengeColor": "white",
    "challengeRank": 15,
    "playerName": player,
    "aggressive": False,
    "endingLength": 4.0,
    "hardwareAccel": False,
    "noteScale": 1.0,
    "fxaa": False,
    "doubleHint": False,
    "disableEffect": False,
    "bitrate": "7M",
    "particle": False,
    "playerRks": 0.0,
    "speed": 1.0,
    "volumeMusic": 1.0,
    "volumeSfx": 1.0,
}
sys.stdout.write(json.dumps({"path": chart_dir, "info": info, "config": config}) + "\n")
sys.stdout.write(json.dumps(os.path.abspath(out_mp4)) + "\n")
