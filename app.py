#!/usr/bin/env python3
"""
CleanVoice Studio — ভিডিও ব্যাকগ্রাউন্ড নয়েজ রিমুভার + ভিডিও এডিটর

ইঞ্জিন (ENGINE env দিয়ে নিয়ন্ত্রণযোগ্য: auto / ai / onnx / rnnoise):
  • ai      → Facebook Denoiser dns48, torch (ফুল কোয়ালিটি, ~1.5GB RAM লাগে)
  • onnx    → dns48-এর ONNX রানটাইম (একই কোয়ালিটি, মাত্র ~400MB — ফ্রি হোস্টিংয়ে চলে!)
  • rnnoise → ffmpeg arnndn (হালকা ফলব্যাক)
'auto' উপলব্ধ সেরাটা বেছে নেয়।

বাকি পাইপলাইন: ffmpeg অডিও বের করা → ডিনয়েজ → ভিডিও এডিট ফিল্টার → রিমাক্স।
"""
import gc
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_file
from scipy import signal

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "outputs"
STATIC_DIR = BASE / "static"
MODELS_DIR = BASE / "models"
for d in (UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, MODELS_DIR):
    d.mkdir(exist_ok=True)

SR = 44100
ONNX_SR = 16000
ONNX_WIN = 80000           # 5s @ 16kHz (frozen ONNX window — ছোট = কম RAM)
CHUNK_SR = 220500          # 5s @ 44.1kHz (= ONNX_WIN রিস্যাম্পলে হুবহু ম্যাপ হয়)
XFADE_IN = 4096
ONNX_MODEL = MODELS_DIR / "dns48_5s.onnx"
RNN_MODEL = MODELS_DIR / "bd.rnnn"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

jobs = {}
jobs_lock = threading.Lock()

ALLOWED_EXT = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp",
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
}

# ----------------------------- engine select -----------------------------

def resolve_engine():
    e = os.environ.get("ENGINE", "auto").strip().lower()
    if e != "auto":
        return e
    try:
        import torch, denoiser  # noqa: F401
        return "ai"
    except Exception:
        pass
    if ONNX_MODEL.exists():
        try:
            import onnxruntime  # noqa: F401
            return "onnx"
        except Exception:
            pass
    if RNN_MODEL.exists():
        return "rnnoise"
    return "ai"


ENGINE = resolve_engine()
print(f"🔧 Engine: {ENGINE}")

# ----------------------------- AI (torch) engine -----------------------------

_model_holder = {"model": None, "sr": 16000, "err": None}
_model_ready = threading.Event()
_model_lock = threading.Lock()


def load_model_bg():
    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        from denoiser import pretrained
        # DENOISER_MODEL env: dns64 (ডিফল্ট, সেরা ব্যালান্স) / dns48 (ফাস্ট) / master64
        name = os.environ.get("DENOISER_MODEL", "dns64")
        m = getattr(pretrained, name, pretrained.dns48)().cpu().eval()
        _model_holder["model"] = m
        _model_holder["sr"] = int(getattr(m, "sample_rate", 16000))
        print(f"✅ Denoiser (torch/{name}) ready, sr={_model_holder['sr']}")
    except Exception as e:  # noqa: BLE001
        _model_holder["err"] = str(e)
        print("❌ Model load failed:", e)
    finally:
        _model_ready.set()


def get_model(jid=None):
    if not _model_ready.is_set():
        if jid:
            update(jid, message="AI মডেল লোড হচ্ছে (প্রথমবার ~১ মিনিট)...")
        _model_ready.wait()
    if _model_holder["err"]:
        raise RuntimeError("AI মডেল লোড করা যায়নি — সার্ভার রিস্টার্ট করুন।")
    return _model_holder["model"], _model_holder["sr"]


# ----------------------------- ONNX engine -------------------------------

_onnx_holder = {"sess": None}
_onnx_lock = threading.Lock()


def get_onnx():
    if _onnx_holder["sess"] is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False   # RAM কম রাখতে (ফ্রি টিয়ারের জন্য জরুরি)
        so.enable_mem_pattern = False
        so.enable_mem_reuse = False
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = max(1, min(2, (os.cpu_count() or 2)))
        so.log_severity_level = 3
        _onnx_holder["sess"] = ort.InferenceSession(str(ONNX_MODEL), so,
                                                    providers=["CPUExecutionProvider"])
        print("✅ Denoiser (ONNX) ready")
    return _onnx_holder["sess"]


# ----------------------------- helpers -----------------------------

def update(jid, **kw):
    with jobs_lock:
        if jid in jobs:
            jobs[jid].update(kw)


def get_job(jid):
    with jobs_lock:
        j = jobs.get(jid)
        return dict(j) if j else None


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path):
    r = sh([FFMPEG, "-hide_banner", "-i", str(path)])
    t = r.stderr or ""
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", t)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return dur, ("Audio:" in t), ("Video:" in t)


def video_info(path):
    """ভিডিওর উচ্চতা ও fps বের করা (AI পলিশ/আপস্কেল/মোশনের সিদ্ধান্তের জন্য)।"""
    r = sh([FFMPEG, "-hide_banner", "-i", str(path)])
    t = r.stderr or ""
    m = re.search(r"Video:.*?\s(\d{2,5})x(\d{2,5})", t)
    h = int(m.group(2)) if m else 0
    fm = re.search(r"([\d.]+)\s*fps", t)
    fps = float(fm.group(1)) if fm else 30.0
    return h, fps


def analyze_video(in_path, ts, te, dur):
    """AI ভিডিও অ্যানালাইসিস: ছোট করে দ্রুত স্যাম্পল নিয়ে গড় উজ্জ্বলতা
    (YAVG), ডাইনামিক রেঞ্জ (YMAX-YMIN) ও রঙের তীব্রতা (SATAVG) মাপা।"""
    span = (te if te > 0 else dur or 60.0) - ts
    span = min(max(span, 1.0), 40.0)
    cmd = [FFMPEG, "-v", "error"]
    if ts > 0:
        cmd += ["-ss", f"{ts:.3f}"]
    cmd += ["-t", f"{span:.3f}", "-i", str(in_path),
            "-vf", "fps=2,scale=320:-2,signalstats,metadata=mode=print:file=-",
            "-an", "-f", "null", "-"]
    r = sh(cmd)
    out = r.stdout or ""

    def avg(key):
        vals = [float(x) for x in re.findall(key + r"=([0-9.]+)", out)]
        return sum(vals) / len(vals) if vals else None

    yavg, ymin, ymax, sat = avg("YAVG"), avg("YMIN"), avg("YMAX"), avg("SATAVG")
    if yavg is None:
        return None
    return {"yavg": yavg, "yrange": (ymax or 235.0) - (ymin or 16.0),
            "satavg": sat if sat is not None else 60.0}


def ai_polish_chain(st):
    """ভিডিওটা নিজে দেখে (analyze) তারপর ঠিকমতো স্মুথ + কালার + শার্প —
    ফিক্সড প্রিসেট নয়, প্রতিটা ক্লিপের জন্য আলাদা টিউনিং!"""
    vf = ["hqdn3d=2.4:2:4:3"]  # গ্রেইন/সেন্সর নয়েজ স্মুদ (ভিডিও সিলকি লাগবে)
    if st:
        bright = max(-0.06, min(0.06, (116.0 - st["yavg"]) / 255.0))
        rng = max(st["yrange"], 60.0)
        contra = max(1.0, min(1.16, 170.0 / rng)) if rng < 170 else 1.04
        sat = max(1.0, min(1.26, 1.05 + (60.0 - st["satavg"]) / 150.0))
        vf.append(f"eq=brightness={bright:.3f}:contrast={contra:.3f}:saturation={sat:.3f}")
    else:
        vf.append("eq=contrast=1.06:saturation=1.10:brightness=0.01")
    vf.append("unsharp=5:5:0.5:5:5:0.0")
    return vf


def fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fmt_size(b):
    try:
        b = int(b)
    except Exception:
        return ""
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024 or u == "GB":
            return f"{b:.1f} {u}" if u != "B" else f"{b} B"
        b /= 1024
    return ""


# ----------------------------- audio denoise -----------------------------

def _denoise_channel_windows(jid, c, nch, chan, sr, mix_amount, enhance_fn,
                             done_ref, total_units):
    """চ্যানেলটা ১০ সেকেন্ডের উইন্ডোতে ভাগ করে enhance_fn প্রয়োগ।"""
    n = len(chan)
    res = np.zeros_like(chan)
    pos = 0
    while pos < n:
        end = min(n, pos + CHUNK_SR)
        seg = chan[pos:end]
        clean = enhance_fn(seg, sr)
        cleaned = mix_amount * clean + (1.0 - mix_amount) * seg
        if pos == 0:
            res[pos:end] = cleaned
        else:
            ov = min(XFADE_IN, end - pos)
            w = np.linspace(0.0, 1.0, ov, dtype=np.float32)
            res[pos:pos + ov] = res[pos:pos + ov] * (1.0 - w) + cleaned[:ov] * w
            res[pos + ov:end] = cleaned[ov:]
        pos = end
        done_ref[0] += 1
        update(jid, progress=8 + int(72 * done_ref[0] / total_units),
               message=f"AI দিয়ে নয়েজ রিমুভ হচ্ছে... ধাপ {done_ref[0]}/{total_units}")
    return res


def denoise_wav_onnx(wav_in, wav_out, jid, mix_amount, bass_cut):
    """স্ট্রিমিং ONNX ডিনয়েজ — ৪ সেকেন্ড স্ট্রাইড + ০.৫ সেকেন্ড কনটেক্সট ওভারল্যাপ।
    কনটেক্সট অংশ মডেলকে প্রসঙ্গ দেয় (LSTM/কনভ এজ-সমস্যা দূর), আউটপুট থেকে ফেলে দেওয়া হয়।
    RAM প্রায় ধ্রুবক থাকে — লম্বা ভিডিওতেও ফ্রি টিয়ারে চলে।"""
    get_onnx()
    sess = _onnx_holder["sess"]
    STEP = CHUNK_SR - 2 * (CHUNK_SR // 10)      # 4s @44.1k = 176400
    CTX = CHUNK_SR // 10                        # 0.5s @44.1k = 22050
    STEP16 = STEP * ONNX_SR // SR               # 64000
    CTX16 = CTX * ONNX_SR // SR                 # 8000

    with sf.SoundFile(str(wav_in)) as fin:
        sr = fin.samplerate
        nch = fin.channels
        total = len(fin)
        if total < 256:
            raise RuntimeError("অডিও পাওয়া যায়নি বা অনেক ছোট!")
        total_steps = max(1, (total + STEP - 1) // STEP)
        total_units = total_steps * nch
        done = 0
        sos = signal.butter(4, 60, "highpass", fs=sr, output="sos") if bass_cut else None
        absmax = 0.0

        with sf.SoundFile(str(wav_out), "w", samplerate=sr, channels=nch,
                          subtype="PCM_16") as fout:
            carry = None
            for step_i, pos in enumerate(range(0, total, STEP)):
                read_from = max(pos - CTX, 0)
                read_len = min(total - read_from, CTX + STEP + CTX)
                fin.seek(read_from)
                block = fin.read(read_len, dtype="float32", always_2d=True)
                seg_len = min(STEP, total - pos)          # এই স্টেপে নতুন অডিও কতটুকু
                pad_pre = pos - read_from                 # বামে প্রকৃত কনটেক্সট (0 বা CTX)

                step16_len = (seg_len * ONNX_SR) // sr
                keep16 = (pad_pre * ONNX_SR) // sr

                out_seg = np.empty((seg_len, nch), dtype=np.float32)
                for c in range(nch):
                    raw = block[:, c].copy()
                    if sos is not None:
                        raw = signal.sosfilt(sos, raw)
                    seg_new = raw[pad_pre:pad_pre + seg_len]
                    src16 = signal.resample_poly(raw, ONNX_SR, sr).astype(np.float32)
                    x = np.zeros((1, 1, ONNX_WIN), np.float32)
                    L16 = min(len(src16), ONNX_WIN)
                    x[0, 0, :L16] = src16[:L16]
                    with _onnx_lock:
                        y16 = sess.run(["y"], {"x": x})[0][0, 0]
                    crop16 = y16[keep16:keep16 + step16_len]
                    back = signal.resample_poly(crop16, sr, ONNX_SR).astype(np.float32)
                    if len(back) >= seg_len:
                        back = back[:seg_len]
                    else:
                        back = np.pad(back, (0, seg_len - len(back)))
                    out_seg[:, c] = mix_amount * back + (1.0 - mix_amount) * seg_new
                    done += 1

                cur = np.concatenate([carry, out_seg]) if carry is not None else out_seg
                if carry is not None:
                    ov = len(carry)
                    w = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
                    cur[:ov] = carry * (1.0 - w) + out_seg[:ov] * w
                last = (pos + seg_len) >= total
                if not last:
                    fout.write(cur[:-XFADE_IN])
                    carry = cur[-XFADE_IN:]
                    wrote = cur[:-XFADE_IN]
                else:
                    fout.write(cur)
                    carry = None
                    wrote = cur
                absmax = max(absmax, float(np.max(np.abs(wrote))))
                del block, out_seg, cur, wrote
                update(jid, progress=8 + int(72 * done / total_units),
                       message=f"AI দিয়ে নয়েজ রিমুভ হচ্ছে... ধাপ {done}/{total_units}")

        if absmax > 0.98:
            g = 0.98 / absmax
            with sf.SoundFile(str(wav_out), "r+") as f:
                while True:
                    mark = f.tell()
                    b = f.read(1 << 20, dtype="float32", always_2d=True)
                    if len(b) == 0:
                        break
                    f.seek(mark)
                    f.write(b * g)
    gc.collect()


def denoise_wav(wav_in, wav_out, jid, mix_amount, bass_cut):
    if ENGINE == "onnx":
        denoise_wav_onnx(wav_in, wav_out, jid, mix_amount, bass_cut)
        return
    data, sr = sf.read(str(wav_in), dtype="float32", always_2d=True)
    n, nch = data.shape
    if n < 256:
        raise RuntimeError("অডিও পাওয়া যায়নি বা অনেক ছোট!")

    sos = signal.butter(4, 60, "highpass", fs=sr, output="sos") if bass_cut and n > 64 else None
    out = np.zeros_like(data)
    per_chan = max(1, (n + CHUNK_SR - 1) // CHUNK_SR)
    total_units = per_chan * nch
    done_ref = [0]

    if ENGINE == "ai":
        model, msr = get_model(jid)
        import torch
        import torchaudio
        res_dn = torchaudio.transforms.Resample(sr, msr)
        res_up = torchaudio.transforms.Resample(msr, sr)

        def enhance_fn(seg, _sr):
            w = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32))[None, None]
            with _model_lock, torch.no_grad():
                o = model(res_dn(w))
            y = res_up(o)[0, 0].numpy()
            if len(y) >= len(seg):
                return y[: len(seg)].astype(np.float32)
            return np.pad(y, (0, len(seg) - len(seg))).astype(np.float32)

    elif ENGINE == "onnx":
        get_onnx()

        def enhance_fn(seg, _sr):
            seg16 = signal.resample_poly(seg, ONNX_SR, sr).astype(np.float32)
            L = len(seg16)
            if L == 0:
                return np.zeros_like(seg)
            x = np.zeros((1, 1, ONNX_WIN), np.float32)
            L2 = min(L, ONNX_WIN)
            x[0, 0, :L2] = seg16[:L2]
            with _onnx_lock:
                y16 = _onnx_holder["sess"].run(["y"], {"x": x})[0][0, 0, :L2]
            back = signal.resample_poly(y16, sr, ONNX_SR).astype(np.float32)
            if len(back) >= len(seg):
                return back[: len(seg)]
            return np.pad(back, (0, len(seg) - len(back)))

    elif ENGINE == "rnnoise":
        raise RuntimeError("internal: rnnoise আলাদা পথে চলে")

    else:
        raise RuntimeError(f"অজানা ইঞ্জিন: {ENGINE}")

    for c in range(nch):
        chan = data[:, c]
        if sos is not None:
            chan = signal.sosfiltfilt(sos, chan).astype(np.float32)
        out[:, c] = _denoise_channel_windows(jid, c, nch, chan, sr, mix_amount,
                                             enhance_fn, done_ref, total_units)
        gc.collect()

    src_peak = float(np.max(np.abs(data))) or 1.0
    out_peak = float(np.max(np.abs(out))) or 1.0
    target = min(src_peak, 0.98)
    if out_peak > target:
        out *= target / out_peak
    sf.write(str(wav_out), out, sr, subtype="PCM_16")


def denoise_wav_rnnoise(wav_in, wav_out, jid, mix_amount, bass_cut):
    """হালকা ফলব্যাক: ffmpeg arnndn (RNNoise)।"""
    update(jid, progress=20, message="নয়েজ রিমুভ হচ্ছে (লাইট ইঞ্জিন)...")
    af = "aresample=48000,arnndn=" + str(RNN_MODEL) + f":mix={mix_amount:.2f},aresample={SR}"
    if bass_cut:
        af = f"highpass=f=60," + af
    r = sh([FFMPEG, "-y", "-loglevel", "error", "-i", str(wav_in),
            "-af", af, "-c:a", "pcm_s16le", str(wav_out)])
    if r.returncode != 0 or not wav_out.exists():
        # mix প্যারাম না চললে পুরনো স্টাইলে
        af2 = "aresample=48000,arnndn=" + str(RNN_MODEL) + f",aresample={SR}"
        r = sh([FFMPEG, "-y", "-loglevel", "error", "-i", str(wav_in),
                "-af", af2, "-c:a", "pcm_s16le", str(wav_out)])
        if r.returncode != 0 or not wav_out.exists():
            raise RuntimeError("নয়েজ রিমুভ করা যায়নি!")
    update(jid, progress=80)


# ----------------------------- voice polish -----------------------------

def _floor_peak(path):
    """ক্লিন wav-এর নয়েজ ফ্লোর (নীচের ১০% ফ্রেম) ও আনুমানিক পিক, dB-তে।"""
    frms = []
    peak = 0.0
    with sf.SoundFile(path) as f:
        while True:
            b = f.read(int(0.25 * SR), dtype="float32", always_2d=True)
            if len(b) == 0:
                break
            frms.append(float(np.sqrt(np.mean(b ** 2))))
            peak = max(peak, float(np.max(np.abs(b))))
    floor = 20 * np.log10(np.percentile(np.asarray(frms) + 1e-12, 10))
    return float(floor), float(20 * np.log10(max(peak, 1e-9)))


def polish_wav(wav_in, jid):
    """স্টুডিও-স্টাইল ভয়েস পলিশ: হালকা কমপ্রেশন (ডাইনামিক্স স্মুদ) +
    স্মার্ট গেইন (কণ্ঠ জোরালো কিন্তু ফ্লোর -48dB-এর উপরে ওঠে না) + লিমিটার।"""
    wav_p = str(OUTPUT_DIR / f"{jid}_polish.wav")
    update(jid, progress=81, message="ভয়েস স্মুদিং/পলিশ হচ্ছে...")
    try:
        floor, peak_db = _floor_peak(wav_in)
    except Exception:
        return str(wav_in)
    # কমপ্রেসরের পর আনুমানিক পিক (threshold -18dB, ratio 2)
    p_out = -18.0 + (peak_db + 18.0) / 2.0 if peak_db > -18 else peak_db
    gain = min(8.0, -52.0 - floor, -2.5 - p_out)   # সর্বোচ্চ +8dB, ফ্লোর≤-52, পিক≤-2.5 (AAC-সেফ)
    gain = max(-6.0, gain)
    af = (f"acompressor=threshold=-18dB:ratio=2:attack=8:release=120,"
          f"volume={gain:.2f}dB,aresample={SR},alimiter=limit=0.794:level=false")
    r2 = sh([FFMPEG, "-y", "-loglevel", "error", "-i", str(wav_in),
             "-af", af, "-ar", str(SR), "-c:a", "pcm_s16le", wav_p])
    if r2.returncode == 0 and os.path.exists(wav_p):
        return wav_p
    return str(wav_in)


# ----------------------------- video edit filters -----------------------------

def build_vfilters(o):
    vf = []
    rot = o["rotate"]
    if rot == 90:
        vf.append("transpose=1")
    elif rot == 270:
        vf.append("transpose=2")
    elif rot == 180:
        vf.append("hflip")
        vf.append("vflip")
    if o["flip"] == "h" and rot != 180:
        vf.append("hflip")
    elif o["flip"] == "v" and rot != 180:
        vf.append("vflip")
    if o.get("ai_polish"):
        vf += ai_polish_chain(o.get("ai_stats"))
    if o["enhance"]:
        vf.append("hqdn3d=2:1.5:3:2")
        vf.append("eq=contrast=1.06:saturation=1.12:brightness=0.01")
        vf.append("unsharp=5:5:0.45:5:5:0.0")
    b = (o["brightness"] - 100.0) / 100.0 * 0.5
    c = o["contrast"] / 100.0
    s = o["saturation"] / 100.0
    if abs(b) > 1e-6 or abs(c - 1.0) > 1e-6 or abs(s - 1.0) > 1e-6:
        vf.append(f"eq=brightness={b:.3f}:contrast={c:.3f}:saturation={s:.3f}")
    if o["sharpen"]:
        vf.append("unsharp=5:5:0.9:5:5:0.0")
    if o.get("upscale") and 0 < o.get("src_h", 9999) < 1000:
        vf.append("scale=1920:-2:flags=lanczos")   # ছোট ভিডিও → ফুল-HD
        vf.append("unsharp=5:5:0.3:5:5:0.0")
    if o["speed"] != 1.0:
        vf.append(f"setpts=PTS/{o['speed']}")
    if o.get("motion") and o.get("src_fps", 30.0) < 55:
        vf.append("minterpolate=fps=60:mi_mode=blend")  # স্মুথ মোশন ৬০fps
    return ",".join(vf)


def atempo_chain(speed):
    parts = []
    s = speed
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s *= 2.0
    parts.append(f"atempo={s}")
    return ",".join(parts)


def needs_reencode(o):
    return (o["rotate"] != 0 or o["flip"] != "none" or o["enhance"] or o["sharpen"]
            or o.get("ai_polish") or o.get("motion") or o.get("upscale")
            or o["speed"] != 1.0 or o["brightness"] != 100 or o["contrast"] != 100
            or o["saturation"] != 100 or o["trim_start"] > 0 or o["trim_end"] > 0)


# ----------------------------- worker -----------------------------

def worker(jid, in_path, mix_amount, bass_cut, polish, o):
    wav_in = UPLOAD_DIR / f"{jid}.wav"
    wav_out = OUTPUT_DIR / f"{jid}.wav"
    try:
        update(jid, state="processing", progress=3, message="ভিডিও বিশ্লেষণ হচ্ছে...")
        dur, has_audio, has_video = probe(in_path)
        if not has_audio:
            raise RuntimeError("এই ভিডিওতে কোনো অডিও ট্র্যাক নেই!")

        ts, te = o["trim_start"], o["trim_end"]
        if te > 0 and te <= ts:
            te = 0.0

        if has_video and (o.get("ai_polish") or o.get("motion") or o.get("upscale")):
            try:
                update(jid, progress=4, message="🤖 AI ভিডিও অ্যানালাইসিস হচ্ছে...")
                o["src_h"], o["src_fps"] = video_info(in_path)
                o["ai_stats"] = analyze_video(in_path, ts, te, dur) if o.get("ai_polish") else None
            except Exception:
                o["ai_stats"] = None

        update(jid, progress=5, message="অডিও আলাদা করা হচ্ছে...")
        cmd = [FFMPEG, "-y", "-loglevel", "error"]
        if ts > 0:
            cmd += ["-ss", f"{ts:.3f}"]
        if te > 0:
            cmd += ["-to", f"{te:.3f}"]
        cmd += ["-i", str(in_path), "-vn", "-ac", "2", "-ar", str(SR),
                "-c:a", "pcm_s16le", str(wav_in)]
        r = sh(cmd)
        if r.returncode != 0 or not wav_in.exists():
            raise RuntimeError("ভিডিও থেকে অডিও বের করা যায়নি!")

        update(jid, progress=7, message="নয়েজ রিমুভ ইঞ্জিন প্রস্তুত হচ্ছে...")
        if ENGINE == "rnnoise":
            denoise_wav_rnnoise(wav_in, wav_out, jid, mix_amount, bass_cut)
        else:
            denoise_wav(wav_in, wav_out, jid, mix_amount, bass_cut)

        final_audio = str(wav_out)
        if polish:
            final_audio = polish_wav(str(wav_out), jid)

        job = get_job(jid) or {}
        stem = re.sub(r"[^\w\-. ]+", "", os.path.splitext(job.get("orig_name", "video"))[0]).strip() or "video"

        reencode = has_video and needs_reencode(o)
        update(jid, progress=84,
               message="ভিডিও এডিট/এনকোড হচ্ছে..." if reencode else "ভিডিওর সাথে ক্লিন অডিও বসানো হচ্ছে...")

        cmd = [FFMPEG, "-y", "-loglevel", "error"]
        if reencode:
            cmd += ["-nostats", "-progress", "pipe:1"]
            if ts > 0:
                cmd += ["-ss", f"{ts:.3f}"]
            if te > 0:
                cmd += ["-to", f"{te:.3f}"]
        cmd += ["-i", str(in_path)]
        if has_video:
            cmd += ["-i", final_audio, "-map", "0:v:0", "-map", "1:a:0"]
            out_path = OUTPUT_DIR / f"{jid}.mp4"
            clean_name = f"{stem}_cleaned.mp4"
            if reencode:
                vf_str = build_vfilters(o)
                if vf_str:
                    cmd += ["-vf", vf_str]
                if o["speed"] != 1.0:
                    cmd += ["-af", atempo_chain(o["speed"])]
                cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p"]
            else:
                cmd += ["-c:v", "copy"]
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", str(out_path)]
        else:
            out_path = OUTPUT_DIR / f"{jid}.m4a"
            clean_name = f"{stem}_cleaned.m4a"
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", final_audio]
            if o["speed"] != 1.0:
                cmd += ["-af", atempo_chain(o["speed"])]
            cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_path)]
            reencode = False

        if reencode:
            span = max(0.1, ((te if te > 0 else dur) - ts)) / max(o["speed"], 0.01)
            expected_us = span * 1e6
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            for line in proc.stdout:
                if line.startswith("out_time_ms="):
                    try:
                        us = int(line.split("=")[1])
                    except ValueError:
                        continue
                    ratio = min(0.999, us / max(expected_us, 1))
                    update(jid, progress=84 + int(15 * ratio),
                           message=f"ভিডিও এডিট/এনকোড হচ্ছে... {int(100 * ratio)}%")
            proc.wait()
            err = proc.stderr.read() if proc.stderr else ""
            rc = proc.returncode
        else:
            r = sh(cmd)
            rc, err = r.returncode, r.stderr

        if rc != 0 or not out_path.exists():
            print("ffmpeg error:", (err or "")[-800:])
            raise RuntimeError("ফাইনাল ফাইল তৈরি করা যায়নি!")

        for p in (wav_in, wav_out, OUTPUT_DIR / f"{jid}_polish.wav"):
            try:
                p.unlink()
            except OSError:
                pass
        gc.collect()
        update(jid, state="done", progress=100, message="সম্পন্ন! 🎉",
               file_url=f"/file/{jid}", dl_url=f"/file/{jid}?dl=1",
               clean_name=clean_name,
               out_size=fmt_size(out_path.stat().st_size),
               duration=round(dur, 1))
    except Exception as e:  # noqa: BLE001
        for p in (wav_in, wav_out, OUTPUT_DIR / f"{jid}_polish.wav"):
            try:
                p.unlink()
            except OSError:
                pass
        gc.collect()
        update(jid, state="error", message=str(e)[:300])


# ----------------------------- routes -----------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(ok=True, engine=ENGINE)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="কোনো ফাইল পাওয়া যায়নি!"), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error="দুঃখিত, এই ফরম্যাটটি সাপোর্ট করে না!"), 400

    strength = min(max(fnum(request.form.get("strength"), 95.0), 0.0), 100.0)
    bass_cut = request.form.get("bass_cut", "1") == "1"
    polish = request.form.get("polish", "1") == "1"

    flip = request.form.get("flip", "none")
    if flip not in ("none", "h", "v"):
        flip = "none"
    try:
        rotate = int(request.form.get("rotate", 0))
    except (TypeError, ValueError):
        rotate = 0
    if rotate not in (0, 90, 180, 270):
        rotate = 0
    speed = fnum(request.form.get("speed"), 1.0)
    speed = min(max(speed, 0.25), 4.0)

    o = {
        "flip": flip,
        "rotate": rotate,
        "enhance": request.form.get("enhance", "0") == "1",
        "sharpen": request.form.get("sharpen", "0") == "1",
        "ai_polish": request.form.get("ai_polish", "1") == "1",
        "motion": request.form.get("motion", "0") == "1",
        "upscale": request.form.get("upscale", "0") == "1",
        "brightness": min(max(fnum(request.form.get("brightness"), 100.0), 40.0), 160.0),
        "contrast": min(max(fnum(request.form.get("contrast"), 100.0), 50.0), 200.0),
        "saturation": min(max(fnum(request.form.get("saturation"), 100.0), 0.0), 250.0),
        "speed": speed,
        "trim_start": max(0.0, fnum(request.form.get("trim_start"), 0.0)),
        "trim_end": max(0.0, fnum(request.form.get("trim_end"), 0.0)),
    }

    jid = uuid.uuid4().hex[:12]
    in_path = UPLOAD_DIR / f"{jid}{ext}"
    f.save(str(in_path))

    # UI slider → AI blend amount (mix 1.0 = পুরো AI আউটপুট = সর্বোচ্চ নয়েজ রিমুভাল)
    # ডিফল্ট ৯৫% → ০.৯৭৭৫ (ফ্লোর প্রায় -৬০dB, শোনা যায় না)
    mix_amount = round(0.55 + 0.45 * (strength / 100.0), 4)
    with jobs_lock:
        jobs[jid] = {
            "state": "queued", "progress": 2, "message": "প্রসেসিং শুরু হচ্ছে...",
            "orig_name": f.filename, "in_size": fmt_size(in_path.stat().st_size),
            "created": time.time(),
        }
    threading.Thread(target=worker, args=(jid, in_path, mix_amount, bass_cut, polish, o),
                     daemon=True).start()
    return jsonify(job_id=jid)


@app.route("/status/<jid>")
def status(jid):
    j = get_job(jid)
    if not j:
        return jsonify(error="Job পাওয়া যায়নি"), 404
    return jsonify(j)


@app.route("/file/<jid>")
def file_serve(jid):
    j = get_job(jid)
    if not j or j.get("state") != "done":
        return jsonify(error="ফাইল এখনো তৈরি হয়নি"), 404
    out_path = OUTPUT_DIR / f"{jid}.mp4"
    if not out_path.exists():
        out_path = OUTPUT_DIR / f"{jid}.m4a"
    if not out_path.exists():
        return jsonify(error="ফাইল পাওয়া যায়নি"), 404
    return send_file(out_path,
                     as_attachment=request.args.get("dl") == "1",
                     download_name=j.get("clean_name", "cleaned.mp4"),
                     conditional=True)


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error="ফাইল অনেক বড়! সর্বোচ্চ ২GB সাপোর্ট করে।"), 413


# ----------------------------- janitor / sample -----------------------------

def janitor():
    while True:
        time.sleep(1800)
        cutoff = time.time() - 7200
        for d in (UPLOAD_DIR, OUTPUT_DIR):
            for p in d.iterdir():
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    pass


def make_sample():
    p = STATIC_DIR / "sample_noisy.mp4"
    if p.exists():
        return
    sh([FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=12:size=640x360:rate=25",
        "-f", "lavfi", "-i",
        "aevalsrc=between(t\\,2\\,10)*0.5*sin(2*PI*(240+70*sin(2*PI*1.5*t))*t):s=44100:d=12",
        "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.30:seed=7:duration=12",
        "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(p)])


# gunicorn-এও (import টাইমে) ব্যাকগ্রাউন্ড সার্ভিস চালু হয়
threading.Thread(target=janitor, daemon=True).start()
threading.Thread(target=make_sample, daemon=True).start()
if ENGINE == "ai":
    threading.Thread(target=load_model_bg, daemon=True).start()
elif ENGINE == "onnx":
    threading.Thread(target=get_onnx, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🎙️  CleanVoice Studio running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
