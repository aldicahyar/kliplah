# Kliplah

Aplikasi pribadi yang mengubah video panjang dari YouTube atau file lokal (podcast, stream, variety show, berita) menjadi klip vertikal 9:16 lengkap dengan hook dan subtitle, siap diunggah ke TikTok, Instagram Reels, dan YouTube Shorts. Semua proses berjalan **lokal** di laptop, tanpa GPU dan tanpa API berbayar. Momen menarik dipilih oleh Claude lewat langganan Anda sendiri (default) atau oleh model lokal lewat Ollama.

Status: **S1 selesai: API + worker.** Backend sudah punya REST API, worker dengan antrean SQLite, dan model data v2 (jenis konten, gaya tiga lapis, klip sebagai daftar potongan). Website (review dan edit klip di browser) menyusul setelah tahap sistem S2–S4.

## Cara kerja singkat

```text
link YouTube / file video
  -> unduh (yt-dlp, + heatmap "paling sering diputar ulang")
  -> transkrip bahasa Indonesia (faster-whisper, CPU, per 10 menit, bisa dilanjutkan)
  -> pilih momen: Claude manual (default, project menunggu balasan Anda)
                  atau Ollama qwen3:4b (otomatis, per jendela 5 menit + pergantian topik + heatmap)
  -> rapikan batas klip (kata -> kalimat -> jeda), crop 9:16 per adegan (MediaPipe), thumbnail
  -> setujui klip -> render (hook + subtitle tertanam, volume -14 LUFS)
```

## Persiapan (Windows)

1. Pasang alat:
   ```powershell
   winget install --id Gyan.FFmpeg -e
   winget install --id Ollama.Ollama -e    # hanya untuk mode Ollama
   winget install --id astral-sh.uv -e
   ollama pull qwen3:4b                     # hanya untuk mode Ollama
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

## Menjalankan API dan worker

```powershell
cd backend
uv run python scripts/dev.py
```

Perintah ini menjalankan API di `http://127.0.0.1:8000` (dokumentasi interaktif di `http://127.0.0.1:8000/docs`) dan worker yang memproses job satu per satu. Hentikan keduanya dengan Ctrl+C. Semua data ada di `DATA_DIR/kliplah.db` dan `DATA_DIR/projects/<id>/`.

Contoh lewat API (PowerShell, pakai `curl.exe`):

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/projects -H "Content-Type: application/json" -d '{\"url\": \"https://www.youtube.com/watch?v=...\", \"content_type\": \"podcast\"}'
curl.exe http://127.0.0.1:8000/api/projects/<project_id>          # status dan progres
curl.exe http://127.0.0.1:8000/api/projects/<project_id>/clips    # daftar klip
curl.exe -X POST http://127.0.0.1:8000/api/clips/<clip_id>/approve # setujui -> render
```

Endpoint lain: unggah file (`POST /api/projects/upload`), edit klip (`PATCH /api/clips/{id}`), aksi massal (`POST /api/clips/bulk`), template gaya (`/api/style-templates`), pengaturan (`/api/system/settings`), dan status sistem (`GET /api/system/health`). Daftar lengkapnya ada di `/docs`.

## Memilih momen dengan Claude (default, tanpa biaya API, tanpa Ollama)

Project baru berhenti setelah transkrip dengan status `awaiting_moments`. Transkrip yang lebih dari 90 menit dipecah menjadi beberapa bagian; setiap bagian dikirim ke Claude dan diimpor terpisah.

**Lewat CLI:**

1. Buat transkrip dan instruksinya:
   ```powershell
   uv run python -m app.cli export-transcript "https://www.youtube.com/watch?v=..."
   ```
   Perintah ini mencetak **Project ID** dan lokasi file `untuk_claude.md` (atau `untuk_claude_part1of2.md`, dan seterusnya).
2. Berikan isi file itu ke Claude, lewat Claude Code ("pilih momen dari file ini") atau tempel ke claude.ai. Simpan balasan JSON-nya ke sebuah file, misalnya `balasan.json`.
3. Buat klipnya:
   ```powershell
   uv run python -m app.cli import-moments <project_id> balasan.json            # satu bagian
   uv run python -m app.cli import-moments <project_id> balasan2.json --part 2  # bagian berikutnya
   ```
   Setelah semua bagian masuk, pilihan Claude dicek dengan aturan yang sama (durasi, duplikat, batas kalimat), lalu klip disiapkan, disetujui, dan dirender (`--no-render` untuk berhenti sebelum render). Untuk menulis ulang file instruksi project yang sudah ada: `export-moments <project_id>`.

**Lewat API:**

```powershell
curl.exe http://127.0.0.1:8000/api/projects/<project_id>/moments-export      # [{part, of, filename, text}]
curl.exe -X POST "http://127.0.0.1:8000/api/projects/<project_id>/moments-import?part=1" --data-binary "@balasan.json"
curl.exe -X POST http://127.0.0.1:8000/api/projects/<project_id>/moments-finish  # lanjut menyiapkan klip
```

Untuk memakai Ollama sebagai default, ubah pengaturan: `PUT /api/system/settings` dengan `{"moment_source": "ollama"}`, atau per perintah CLI dengan `--moments ollama`.

## Perintah CLI lain

```powershell
cd backend
uv run python -m app.cli run "https://www.youtube.com/watch?v=..."               # proses; dengan Claude manual berhenti di file ekspor
uv run python -m app.cli run "D:\rekaman\episode1.mp4" --moments ollama --no-render
uv run python -m app.cli analyze <project_id> --llm-model gemma3:4b              # pilih momen ulang dengan Ollama
uv run python -m app.cli bench-whisper "<link atau file>" --models small medium --seconds 300
uv run python -m app.cli migrate-m1    # pindahkan hasil CLI M1 lama ke database (sekali, aman diulang)
uv run python -m app.cli recover       # kembalikan job yang tertinggal ke antrean
```

CLI menulis ke database yang sama dengan API, dan tanpa opsi tambahan menjalankan job project-nya sendiri secara langsung.

- **Saat `scripts/dev.py` sedang berjalan**, tambahkan `--queue-only` (untuk `run`, `export-transcript`, `analyze`, `import-moments`) supaya CLI hanya mengantrekan job dan worker yang memprosesnya.
- **`recover` hanya untuk saat tidak ada worker yang berjalan.** Perintah ini mengembalikan job `running` dan klip `rendering` yang tertinggal (misalnya karena laptop mati) ke antrean. Jika dijalankan saat `dev.py` aktif, job yang sedang dikerjakan worker ikut dianggap tertinggal. Worker sendiri melakukan pemulihan ini otomatis setiap kali menyala.

Hasil render ada di `DATA_DIR/projects/<id>/clips/<clip_id>/`: `final.mp4` dan `captions.txt` (judul, caption, dan hashtag per platform). Semua tahap menyimpan checkpoint, jadi proses yang terhenti (laptop mati, Ollama tertutup) dilanjutkan dari tahap terakhir. Error sementara diulang otomatis sampai 3 kali.

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
