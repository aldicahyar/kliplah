# Kliplah

Aplikasi pribadi yang mengubah video panjang dari YouTube atau file lokal (podcast, stream, variety show, berita) menjadi klip vertikal 9:16 lengkap dengan hook dan subtitle, siap diunggah ke TikTok, Instagram Reels, dan YouTube Shorts. Semua proses berjalan **lokal** di laptop, tanpa GPU dan tanpa API berbayar. Momen menarik dipilih oleh Claude lewat langganan Anda sendiri (default) atau oleh model lokal lewat Ollama.

Status: **S3 selesai: bahasa asing, terjemahan, & subtitle dua bahasa.** Backend sudah mendukung `probe`, unduhan per rentang waktu, cookies browser, subtitle YouTube lintas bahasa, Whisper multibahasa dengan deteksi bahasa otomatis dan keyakinan per kata (`prob`), terjemahan Indonesia lewat Ollama atau Claude manual (`export-translation` / `import-translation`), mode tampilan subtitle (`target`, `bilingual`, `original`), sinyal live chat + puncak suara, arahan `focus_prompt`, profil konten, dan model data v2. Website (review dan edit klip di browser) menyusul setelah tahap sistem S4.

## Cara kerja singkat

```text
link YouTube / file video (+ opsional: --range, --focus, --source-language, --target-language, --transcript-source)
  -> probe (durasi, subtitle, live chat, hint bahasa, rencana subtitle, perkiraan disk & waktu)
  -> unduh (yt-dlp penuh 1080p tanpa encode ulang, atau per rentang maks 720p + encode cepat ke source_r<i>.mp4, opsional cookies_browser)
  -> audio & jeda + transkrip (urutan auto: subtitle resmi target -> source_language -> en -> abjad pertama -> faster-whisper CPU multibahasa)
  -> terjemah (tahap translate ke translation.json bila bahasa sumber != target dan mode Ollama; untuk Claude manual, klip terpilih diterjemahkan lewat lines / export-translation)
  -> sinyal (heatmap, puncak live chat, puncak suara ebur128)
  -> pilih momen: Claude manual (default, project menunggu balasan Anda)
                  atau Ollama qwen3:4b (otomatis, memakai teks terjemahan bila asing + profil konten + focus_prompt + bonus sinyal)
  -> rapikan batas klip (kata -> kalimat -> jeda), crop 9:16 per adegan (MediaPipe), subtitle dua bahasa (text + text_original), thumbnail, proxy
  -> setujui klip -> render (hook + subtitle tertanam sesuai language_mode, volume -14 LUFS)
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

Endpoint lain: cek link sebelum proses (`POST /api/projects/probe`), unggah file (`POST /api/projects/upload`), ekspor/impor terjemahan Claude (`GET /api/projects/{id}/translation-export`, `POST /api/projects/{id}/translation-import`), edit klip (`PATCH /api/clips/{id}`), aksi massal (`POST /api/clips/bulk`), template gaya (`/api/style-templates`), daftar bahasa (`GET /api/system/languages`), pengaturan (`/api/system/settings`, termasuk `cookies_browser`, `encode_threads`, `range_max_height`), dan status sistem (`GET /api/system/health`). Daftar lengkapnya ada di `/docs`.

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

## Video berbahasa asing dan terjemahan

- **Deteksi dan pilihan bahasa (`--source-language` / `source_language`):** kosong/`auto` (default) membuat Whisper mendeteksi bahasa dari potongan audio pertama yang memiliki ≥ 10 kata. Hasil deteksi Melayu (`ms`) dianggap bahasa target Indonesia (`id`), dan keyakinan `< 0,6` otomatis kembali ke bahasa target. Setiap kata menyimpan keyakinan `prob`. Daftar 19 kode bahasa tersedia di `GET /api/system/languages`.
- **Urutan sumber subtitle (`--transcript-source auto`):** untuk YouTube, `auto` memilih subtitle resmi (`youtube_manual`) dengan urutan prioritas: bahasa target (`id`) → `source_language` (jika diisi dan tersedia) → `en` → bahasa pertama menurut abjad. Jika tidak ada subtitle resmi, otomatis memakai `whisper`. `youtube_auto` tidak pernah dipilih otomatis dan hanya dipakai bila dipilih eksplisit (`--transcript-source youtube_auto`). File unggahan lokal selalu memakai `whisper`.
- **Contoh bahasa asing dengan Claude** *(dianjurkan untuk Korea, Jepang, Mandarin, Thailand)*:
  ```powershell
  uv run python -m app.cli run "https://www.youtube.com/watch?v=..." --source-language ko --moments claude
  ```
  Claude membaca transkrip bahasa asli di `untuk_claude*.md`, menulis hook/alasan/judul/caption/hashtag dalam bahasa Indonesia, dan diminta menyertakan field `"lines": [{"n": ..., "text": "..."}]` untuk kalimat di sekitar klip terpilih (`start_sentence - 1` sampai `end_sentence + 1`). Saat diimpor, terjemahan `lines` langsung disimpan ke `translation.json` (balasan tanpa `lines` tetap diterima dan bisa diterjemahkan lewat langkah di bawah).
- **Alur terjemahan manual terpisah (`export-translation` / `import-translation`):** dipakai bila balasan momen belum menyertakan `lines`, masih ada kalimat klip yang belum diterjemahkan, atau ingin menerjemahkan seluruh transkrip:
  ```powershell
  uv run python -m app.cli export-translation <project_id>                      # default --scope clips (atau --scope all)
  # Berikan isi terjemahan_claude*.md ke Claude, simpan JSON balasannya ke terjemahan.json
  uv run python -m app.cli import-translation <project_id> terjemahan.json      # tambah --part 2 bila > 300 baris
  ```
  Lewat API: `GET /api/projects/<project_id>/translation-export?scope=clips` lalu `POST /api/projects/<project_id>/translation-import?part=1` (body teks mentah UTF-8). Impor memperbarui `translation.json` dan menyegarkan subtitle klip yang belum diedit manual (klip yang sudah dirender ditandai `needs_rerender`; klip dengan baris `edited` dilewati).
- **Mode Ollama (`--moments ollama`):** bila bahasa sumber berbeda dari target, tahap `translate` menerjemahkan transkrip dalam batch 40 kalimat ke `translation.json` (dengan checkpoint per batch) sebelum analisis momen. Untuk `ko`, `ja`, `zh`, dan `th`, `probe` memberi peringatan agar mempertimbangkan jalur Claude manual.
- **Mode tampilan subtitle (`subtitle.language_mode`):** diatur lewat gaya project (`PATCH /api/projects/{id}`) atau per klip (`PATCH /api/clips/{id}` pada `style_override`):
  - `target` *(default)*: hanya menampilkan teks target (Indonesia).
  - `bilingual`: menampilkan teks asli (`text_original`) di atas teks terjemahan dengan skala ukuran font `0,72×`.
  - `original`: menampilkan teks bahasa asli (`text_original`, atau fallback ke `text` bila kosong).
  Mode **Sorot kata** (`highlight_word`) otomatis turun menjadi tampilan per baris jika kata berwaktu perkiraan (`approx: true`) atau `language_mode` bukan `target`.

## Perintah CLI lain

```powershell
cd backend
uv run python -m app.cli probe "https://www.youtube.com/watch?v=..." --type stream --source-language en --range 00:30:00-01:00:00
uv run python -m app.cli config set cookies_browser chrome                       # untuk video dibatasi umur
uv run python -m app.cli config set encode_threads 4                             # jumlah thread CPU saat encode rentang
uv run python -m app.cli config set range_max_height 720                         # resolusi maks rentang (480/720/1080)
uv run python -m app.cli run "https://www.youtube.com/watch?v=..." --type stream --range 05:20:00-05:50:00 --focus "reaksi pertama masuk dunia GTA 6" --moments ollama --no-render
uv run python -m app.cli run "https://www.youtube.com/watch?v=..." --source-language ko --moments claude --no-render
uv run python -m app.cli run "D:\rekaman\episode1.mp4" --moments ollama --no-render
uv run python -m app.cli export-translation <project_id> --scope clips
uv run python -m app.cli import-translation <project_id> terjemahan.json --part 1
uv run python -m app.cli analyze <project_id> --llm-model gemma3:4b              # pilih momen ulang dengan Ollama
uv run python -m app.cli bench-whisper "<link atau file>" --models small medium --seconds 300
uv run python -m app.cli migrate-m1    # pindahkan hasil CLI M1 lama ke database (sekali, aman diulang)
uv run python -m app.cli recover       # kembalikan job yang tertinggal ke antrean
```

CLI menulis ke database yang sama dengan API, dan tanpa opsi tambahan menjalankan job project-nya sendiri secara langsung.

- **Unduhan per rentang (`--range`):** setiap rentang diunduh maksimal `720p` (`range_max_height`: `480`, `720`, atau `1080`) lalu di-encode cepat (`libx264 ultrafast`, output klip akhir tetap `1080x1920`) supaya batas waktu tepat tanpa membebani CPU laptop. Jumlah thread CPU diatur lewat `encode_threads` (default `cpu_count - 2`, minimal `1`, mis. `config set encode_threads 4`). Video penuh tetap diunduh sesuai `MAX_VIDEO_HEIGHT` (`1080p`) tanpa encode ulang.

- **Saat `scripts/dev.py` sedang berjalan**, tambahkan `--queue-only` (untuk `run`, `export-transcript`, `analyze`, `import-moments`) supaya CLI hanya mengantrekan job dan worker yang memprosesnya.
- **`recover` hanya untuk saat tidak ada worker yang berjalan.** Perintah ini mengembalikan job `running` dan klip `rendering` yang tertinggal (misalnya karena laptop mati) ke antrean. Jika dijalankan saat `dev.py` aktif, job yang sedang dikerjakan worker ikut dianggap tertinggal. Worker sendiri melakukan pemulihan ini otomatis setiap kali menyala.

Hasil render ada di `DATA_DIR/projects/<id>/clips/<clip_id>/`: `final.mp4` dan `captions.txt` (judul, caption, dan hashtag per platform). Semua tahap menyimpan checkpoint, jadi proses yang terhenti (laptop mati, Ollama tertutup) dilanjutkan dari tahap terakhir. Error sementara diulang otomatis sampai 3 kali.

## Batas yang diketahui

- **Transkrip campuran bahasa:** satu project memakai satu bahasa sumber (`source_language`), misalnya video berbahasa Indonesia yang menyelipkan frasa atau kalimat Inggris. Perbaikan kata/baris lewat kamus koreksi dan editor subtitle di browser ditunda ke W2.
- **Kualitas terjemahan lokal pada bahasa tertentu:** uji video asli (Step 4, `qwen3:4b` CPU) menunjukkan bahasa dengan penanda `weak_local_translation` (`ko`, `ja`, `zh`, `th`) berjalan jauh lebih lambat (~3 menit proses per menit video, `TRANSLATE_SEC_PER_MIN = 190.0`) dan lebih rentan salah terjemah nama/makanan khas dibanding `claude_manual`; gunakan jalur Claude manual untuk kualitas terbaik.

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
