# 🚀 CleanVoice Studio — ফ্রিতে ডিপ্লয় গাইড (কোনো কার্ড লাগে না!)

## ✅ রেকমেন্ডেড: Render.com ফ্রি টিয়ার

**কেন এটা**: কার্ড লাগে না, GitHub অ্যাকাউন্টেই সাইনআপ, অ্যাপের AI মডেল ৫১২MB ফ্রি RAM-এ চলে।

ফ্রি টিয়ারের শর্ত:
- 512MB RAM, মাসে ৭৫০ ঘণ্টা ফ্রি রানটাইম
- ১৫ মিনিট কেউ ব্যবহার না করলে ঘুমিয়ে যায় → পরের ভিজিটে জাগতে ~৩০-৬০ সেকেন্ড
- আপনার লিংক: `https://your-app.onrender.com`

### ধাপ ১: GitHub অ্যাকাউন্ট (ফ্রি)
1. [github.com/join](https://github.com/join) → ইমেইল দিয়ে সাইন আপ (কার্ড লাগে না)

### ধাপ ২: কোড আপলোড
1. লগিন করে ডান-উপরে **+** → **New repository**
2. নাম: `cleanvoice-studio` → **Public** রেখে → **Create repository**
3. পেজে **"uploading an existing file"** লিংকে ক্লিক (বা **Add file → Upload files**)
4. ZIP আনজিপ করে **সব ফাইল + ফোল্ডার ড্র্যাগ করুন** (git ছাড়াই ব্রাউজার থেকে!):
```
├── app.py
├── Dockerfile             ← VPS/Oracle-এর জন্য (ফুল torch AI)
├── Dockerfile.render      ← Render-এর জন্য (স্লিম ONNX)
├── render.yaml            ← Render অটো-কনফিগ
├── export_onnx.py         ← বিল্ডের সময় মডেল তৈরি করে
├── requirements.txt
├── requirements-slim.txt
├── README.md
├── .dockerignore
├── DEPLOY.md
├── models/bd.rnnn
├── templates/index.html
└── static/sample_noisy.mp4
```
5. সবুজ **Commit changes** বাটন

### ধাপ ৩: Render অ্যাকাউন্ট
1. [render.com](https://render.com) → **Sign up with GitHub** (কার্ড লাগে না!)
2. GitHub অ্যাক্সেস অনুমতি দিন

### ধাপ ৪: ডিপ্লয় (Blueprint — এক ক্লিকে সব কনফিগ)
1. Render Dashboard → **New +** → **Blueprint**
2. `cleanvoice-studio` রিপো সিলেক্ট → **Connect**
3. render.yaml নিজে নিজে পড়ে সব সেট করবে → **Deploy** / **Apply**
4. ৫-৮ মিনিট বিল্ড (AI মডেল ONNX-এ এক্সপোর্ট হয় এই সময়ে) → **Live** 🎉

**অথবা ম্যানুয়ালি**: New + → **Web Service** → রিপো Connect →
Runtime: **Docker**, Dockerfile Path: `./Dockerfile.render`, Instance Type: **Free** → Deploy

### ধাপ ৫: লাইভ!
- ড্যাশবোর্ডে আপনার URL পাবেন: `https://cleanvoice-studio-xxxx.onrender.com`
- প্রথম ভিজিটে Space ঘুম থেকে জাগতে ~৩০-৬০ সেকেন্ড নিতে পারে (ফ্রি টিয়ারের নিয়ম)

### ডোমেইন লাগলে (অপশনাল)
- Render Dashboard → Settings → Custom Domain → নিজের ডোমেইন যোগ (DNS রেকর্ড দেবে)

---

## বিকল্প পথগুলো

<details>
<summary><b>Oracle Cloud Always Free</b> (4 CPU, 24GB RAM — কার্ড ভেরিফিকেশন লাগে)</summary>

1. cloud.oracle.com → Sign up (ডুয়াল-কারেন্সি কার্ড লাগে, চার্জ হয় না)
2. VM.Standard.A1.Flex (Always Free) — 4 OCPU / 24GB
3. `curl -fsSL https://get.docker.com | sh`
4. `docker build -t cleanvoice . && docker run -d --restart always -p 80:7860 cleanvoice`
   (মূল `Dockerfile` ব্যবহার করুন — ফুল torch AI ইঞ্জিন চলবে)
5. Security List-এ port 80 খুলুন
</details>

<details>
<summary><b>বাংলাদেশি VPS</b> (bKash পেমেন্ট, ~৳৫০০-১০০০/মাস)</summary>

ExonHost / XeonBD / DianaHost থেকে KVM VPS (2 vCPU, 4GB RAM, Ubuntu 22.04) নিয়ে
উপরের Docker কমান্ড ২টাই যথেষ্ট।
</details>

## সমস্যা হলে

- Render লগে **"Out of memory"** দেখালে: Dashboard → Environment → `ENGINE=rnnoise` সেট করুন
  (আরও ছোট RNNoise ইঞ্জিনে চলবে — রিবিল্ড লাগে না, শুধু Restart)
- বিল্ড ফেইল করলে Logs ট্যাবে এরর কপি করে রাখুন
- `/health` এন্ডপয়েন্টে ইঞ্জিন স্ট্যাটাস দেখা যায়
