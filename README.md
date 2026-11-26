# Kliplah

Aplikasi pribadi yang mengubah podcast YouTube berbahasa Indonesia menjadi klip vertikal 9:16 lengkap dengan hook dan subtitle, siap diunggah ke TikTok, Instagram Reels, dan YouTube Shorts. Semua AI berjalan **lokal dan gratis** di laptop (faster-whisper + Ollama), tanpa GPU dan tanpa API berbayar.

Status: **M1, pipeline lewat command line.** Website (review dan edit klip di browser) sedang dikembangkan.

## Cara kerja singkat

```text
link YouTube / file video
  -> unduh (yt-dlp, + heatmap "paling sering diputar ulang")
  -> transkrip bahasa Indonesia (faster-whisper, CPU, per 10 menit, bisa dilanjutkan)
  -> cari momen menarik (Ollama qwen3:4b per jendela 5 menit + pergantian topik + heatmap)
  -> rapikan batas klip (kata -> kalimat -> jeda)
  -> crop 9:16 per adegan mengikuti pembicara (MediaPipe)
  -> render (hook + subtitle tertanam, volume -14 LUFS) + laporan evaluasi
```

## Persiapan (Windows)

1. Pasang alat:
   ```powershell
   winget install --id Gyan.FFmpeg -e
   winget install --id Ollama.Ollama -e
   winget install --id astral-sh.uv -e
   ollama pull qwen3:4b
   ```
2. Simpan file besar di luar OneDrive. Salin `backend/.env.example` ke `backend/.env` dan atur `DATA_DIR`. Lalu buat virtual env di luar OneDrive:
   ```powershell
   $env:UV_PROJECT_ENVIRONMENT = "C:/Users/<nama>/.venvs/klipper"
   cd backend
   uv sync
   ```
3. Cek kesiapan:
   ```powershell
   uv run python scripts/check_setup.py
   ```
   Saat pertama kali dijalankan, model deteksi wajah MediaPipe (sekitar 3,6 MB) diunduh otomatis ke `DATA_DIR/models`.

## Pemakaian

```powershell
cd backend
uv run python -m app.cli run "https://www.youtube.com/watch?v=..."     # proses penuh + render
uv run python -m app.cli run "D:\rekaman\episode1.mp4" --no-render      # file lokal, tanpa render
uv run python -m app.cli analyze <project_id> --llm-model gemma3:4b     # bandingkan model LLM
uv run python -m app.cli bench-whisper "<link atau file>" --models small medium --seconds 300
```

### Opsi A: Claude memilih momen lewat langganan Anda (tanpa biaya API, tanpa Ollama)

1. Buat transkrip dan instruksinya:
   ```powershell
   uv run python -m app.cli export-transcript "https://www.youtube.com/watch?v=..."
   ```
   Perintah ini mencetak **Project ID** dan lokasi file `untuk_claude.md`.
2. Berikan isi `untuk_claude.md` ke Claude, lewat Claude Code ("pilih momen dari file ini") atau tempel ke claude.ai. Simpan balasan JSON-nya ke sebuah file, misalnya `balasan.json`.
3. Buat klipnya:
   ```powershell
   uv run python -m app.cli import-moments <project_id> balasan.json
   ```
   Pilihan Claude tetap dicek dengan aturan yang sama (durasi, duplikat, batas kalimat), lalu di-crop, diberi subtitle, dan dirender. Hasilnya ada di folder `runs/<whisper>__claude-manual/`.

Hasilnya ada di `DATA_DIR/projects/<id>/runs/<whisper>__<llm>/`:
- `clips/<rank>_<id>/final.mp4` dan `captions.txt` (judul, caption, dan hashtag per platform)
- `report.md`: isi kolom **Layak? (ya/tidak)** untuk gerbang evaluasi M1. Laporan yang sudah diisi tidak pernah ditimpa; run berikutnya menulis `report-<waktu>.md`.

Semua tahap menyimpan checkpoint, jadi perintah yang terhenti (laptop mati, Ollama tertutup) bisa diulang dan akan melanjutkan dari tahap terakhir.

## Pengujian

```powershell
cd backend
uv run pytest -q                      # test cepat
uv run pytest -q -m "slow or not slow" # termasuk test dengan ffmpeg dan MediaPipe asli
```

## Catatan hukum

- Mengunduh video YouTube melanggar Ketentuan Layanan YouTube. Aplikasi ini dibuat untuk **pemakaian pribadi**.
- Mengunggah ulang potongan konten milik orang lain dapat terkena klaim hak cipta di platform tujuan. **Tanggung jawabnya ada pada pengguna.** Mintalah izin kreator aslinya, atau gunakan konten milik sendiri.

## Kredit

- Font **Montserrat** (The Montserrat Project Authors), lisensi SIL Open Font License 1.1, lihat `backend/assets/fonts/OFL.txt`.
- Beberapa teknik dan nilai parameter terinspirasi dari project berlisensi MIT: [fralapo/clippyme](https://github.com/fralapo/clippyme) (perapian batas klip, crop per adegan), [ClipsAI/clipsai](https://github.com/ClipsAI/clipsai) (deteksi pergantian topik, pemilihan pembicara), dan [Anil-matcha/AI-Youtube-Shorts-Generator](https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator) (kriteria momen, parsing JSON). Seluruh kode di repo ini ditulis sendiri.
