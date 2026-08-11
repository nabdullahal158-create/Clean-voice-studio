# 💻 লোকালি চালানোর গাইড (নিজের কম্পিউটারে)

## Windows

1. **Python ইনস্টল**: [python.org](https://python.org/downloads) থেকে Python 3.11/3.12
   ডাউনলোড → ইনস্টলারে **"Add python.exe to PATH"** টিক দিন (খুব জরুরি!) → Install

2. **এই ফোল্ডারে** Terminal/CMD খুলুন (ফোল্ডারে গিয়ে address bar-এ `cmd` লিখে Enter)

3. **লাইব্রেরি ইনস্টল** (একবারই লাগবে):
```cmd
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

4. **চালু করুন**:
```cmd
python app.py
```

5. **ব্রাউজারে খুলুন** → http://localhost:8000 🎉

## Linux / Mac

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

## জানা ভালো

- প্রথম রানে AI মডেল (~128MB) নিজে ডাউনলোড হবে — ইন্টারনেট লাগবে (একবারই)
- কমপক্ষে **4GB RAM** রেকমেন্ডেড (8GB হলে ভালো)
- GPU ছাড়াই চলে (CPU-তেই দারুণ স্পিড)
- FFmpeg আলাদা করে ইনস্টল করতে হবে **না** (প্যাকেজের সাথেই আসে)

## ঐচ্ছিক সেটিংস (environment variable)

```
set DENOISER_MODEL=dns64    :: ডিফল্ট — সেরা নয়েজ রিমুভাল (আমাদের বেঞ্চমার্ক-প্রমাণিত)
set DENOISER_MODEL=dns48    :: ১.৫x দ্রুত, কোয়ালিটি প্রায় একই
set DENOISER_MODEL=master64 :: স্পিচ সবচেয়ে ন্যাচারাল, একটু ধীর
set ENGINE=onnx             :: কম RAM-এর পুরনো PC-তে (models/ ফোল্ডারে dns48_5s.onnx লাগবে)
```
