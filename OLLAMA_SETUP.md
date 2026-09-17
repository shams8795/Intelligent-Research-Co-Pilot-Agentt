# Ollama Setup - One-time Configuration

## الخطوة 1: تحميل Ollama
- اذهب لـ https://ollama.com/download
- حمل النسخة المناسبة لجهازك (Windows)
- اتبع خطوات التثبيت

## الخطوة 2: تحميل النموذج (مرة واحدة فقط)
افتح PowerShell وشغل:
```powershell
ollama pull mistral
```
هذا يحمل النموذج (~4GB) - تشغيل واحد فقط!

## الخطوة 3: شغّل Ollama Server
افتح PowerShell **جديد** وشغل:
```powershell
ollama serve
```
هذا يشغل Ollama في الخلفية على http://localhost:11434

## الخطوة 4: شغّل التطبيق
افتح PowerShell **آخر** (ثالث) وشغل:
```powershell
cd "c:\Users\win 11\Downloads\Intelligent Research Co-Pilot Agent"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## 📌 النقاط المهمة:
- ✅ Ollama يحتاج 3 terminals منفصلة:
  1. Terminal 1: `ollama serve` (يبقى شغّال)
  2. Terminal 2: `streamlit run app.py` (يبقى شغّال)
  3. Terminal 3: للـ debugging إذا لزم

- ✅ المرة الأولى فقط: `ollama pull mistral` (يحمل النموذج)
- ✅ الباقي: كل مرة افتح Terminal جديد وشغل الأوامر أعلاه

- ⚠️ Ollama يحتاج محرك وحدة معالجة الرسومات (GPU) مثالي:
  - GPU NVIDIA ← الأسرع
  - GPU AMD ← يعمل
  - CPU فقط ← يعمل لكن بطيء جداً

## 📊 الأداء المتوقع:
- ملخص واحد: 60-120 ثانية (حسب جهازك)
- بدون Groq rate limits ✨
- بدون تكاليف API 💰

---

**الآن شغّل المشروع وجرّب!** 🚀
