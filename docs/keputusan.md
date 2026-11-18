# Keputusan Proyek

Dokumen ini mencatat keputusan metodologis dan teknis yang diambil selama pengembangan proyek **Prediksi Peristiwa Geopolitik Global: Analisis Dampak Terhadap Nilai Tukar**. Setiap keputusan disertai alasan pemilihan serta alternatif yang dipertimbangkan agar proses penelitian dapat ditelusuri dan diaudit.

---

## 1. Pemilihan Sumber Data

### Keputusan
Menggunakan **CNBC** sebagai satu-satunya sumber berita geopolitik dan **JISDOR Bank Indonesia** sebagai sumber nilai tukar.

### Alasan
- CNBC menyediakan berita internasional yang relevan dengan peristiwa geopolitik dan memiliki keterkaitan dengan ekonomi serta pasar keuangan.
- Struktur artikel CNBC relatif konsisten sehingga mempermudah proses scraping dan preprocessing.
- JISDOR merupakan kurs referensi harian USD/IDR yang diterbitkan langsung oleh Bank Indonesia.
- Penggunaan satu sumber berita mengurangi kebutuhan penyelarasan skema dari beberapa sumber.
- Ketentuan tugas menetapkan Bank Indonesia sebagai sumber data nilai tukar.

### Alternatif yang Dipertimbangkan
Menggabungkan beberapa sumber berita untuk memperluas cakupan data.

### Alasan Tidak Dipilih
Penggabungan beberapa sumber akan menambah kompleksitas preprocessing, penyamaan format, deduplikasi lintas sumber, serta penyelarasan timestamp.

---

## 2. Pemilihan Metode Akuisisi CNBC

### Keputusan
Menggunakan **Playwright** untuk mengambil indeks artikel CNBC dari halaman archive.

### Alasan
Halaman archive CNBC menggunakan JavaScript untuk memuat dan merender elemen artikel. Playwright memungkinkan halaman dijalankan seperti browser sehingga elemen artikel dan tautannya dapat diidentifikasi.

### Alternatif yang Dipertimbangkan
Menggunakan HTTP request langsung atau melakukan scraping berdasarkan kategori politik.

### Alasan Tidak Dipilih
- Request langsung tidak selalu memperoleh elemen yang dirender secara dinamis.
- Halaman kategori politik CNBC tidak menyediakan seluruh artikel hingga periode yang dibutuhkan, terutama untuk artikel lama.
- Halaman archive memberikan pendekatan berbasis tanggal yang lebih sesuai dengan rentang waktu penelitian.

---

## 3. Pemilihan Strategi Filtering Berita

### Keputusan
Melakukan filtering dalam dua tahap:

1. Filtering berdasarkan judul.
2. Filtering berdasarkan konten artikel setelah full scraping.

### Alasan
Jumlah awal artikel sangat besar, yaitu **96.558 artikel**, sehingga mengambil isi penuh seluruh artikel sejak awal tidak efisien.

Filtering judul menggunakan lexicon geopolitik dapat mengurangi jumlah artikel yang perlu di-*fetch* sebelum proses scraping penuh.

### Alternatif yang Dipertimbangkan
Melakukan scraping isi penuh terhadap seluruh artikel kemudian melakukan klasifikasi relevansi.

### Alasan Tidak Dipilih
Pendekatan tersebut membutuhkan jauh lebih banyak request, waktu scraping, dan sumber daya komputasi.

---

## 4. Pemilihan Lexicon-Based Filtering

### Keputusan
Menggunakan **keyword/lexicon geopolitik** untuk tahap awal filtering judul.

### Alasan
- Sederhana dan mudah diaudit.
- Tidak membutuhkan model machine learning tambahan.
- Cepat diterapkan pada puluhan ribu artikel.
- Keyword dapat disesuaikan dengan ruang lingkup penelitian.

### Validasi
Metode diuji menggunakan audit manual terhadap **50 sampel acak** dan menghasilkan presisi sebesar **86%**.

### Catatan
Filtering berbasis lexicon digunakan sebagai tahap penyaringan awal, bukan sebagai klaim bahwa seluruh artikel yang lolos pasti relevan.

---

## 5. Pemilihan Sumber Utama Ekstraksi Artikel

### Keputusan
Menggunakan **JSON-LD sebagai sumber utama** untuk ekstraksi metadata dan isi artikel.

Urutan ekstraksi yang digunakan:

1. JSON-LD
2. Meta tag
3. CSS selector sebagai fallback

### Alasan
JSON-LD merupakan data terstruktur yang lebih stabil terhadap perubahan tampilan halaman dibandingkan selector CSS.

### Alasan Menggunakan Fallback
Struktur artikel CNBC dapat berbeda antara artikel lama dan baru. Oleh karena itu, meta tag dan CSS selector tetap disediakan untuk menangani artikel yang tidak lengkap pada lapisan sebelumnya.

---

## 6. Validasi CSS Selector

### Keputusan
Sebuah kandidat container artikel dari CSS selector hanya dianggap valid apabila mengandung minimal **3 elemen paragraf**.

### Alasan
Beberapa elemen halaman seperti widget "artikel terkait" dapat memiliki struktur yang menyerupai isi artikel.

Ambang minimal tiga paragraf digunakan untuk mengurangi risiko mengambil konten yang bukan merupakan badan artikel.

---

## 7. Penggunaan Checkpoint pada Scraping

### Keputusan
Menggunakan sistem checkpoint selama proses scraping dan menyimpan progres setiap **200 artikel**.

### Alasan
Scraping artikel merupakan proses yang panjang dan dapat terhenti karena error, koneksi, atau penghentian manual.

Checkpoint memungkinkan:
- proses dihentikan tanpa kehilangan seluruh progres;
- proses dilanjutkan tanpa mengulang artikel yang sudah selesai;
- hasil scraping dipulihkan dari checkpoint sebelumnya.

### Keputusan Tambahan
URL yang sudah selesai diproses dibaca kembali dari seluruh file checkpoint untuk menentukan artikel yang masih harus diproses.

---

## 8. Pemisahan Folder Checkpoint Produksi dan Validasi

### Keputusan
Eksperimen validasi menggunakan folder checkpoint yang terpisah dari proses produksi.

### Alasan
Pada akhir proses, seluruh file checkpoint dalam folder yang dirujuk akan digabungkan. Jika validasi menggunakan folder produksi, hasil scraping sebelumnya dapat ikut masuk ke dataset validasi.

### Parameter Isolasi
Tiga parameter disediakan:

- `--index-file`
- `--out-csv`
- `--ckpt-dir`

Keputusan ini memastikan eksperimen validasi dapat dijalankan tanpa mencemari hasil produksi.

---

## 9. Tidak Menggunakan `--limit-articles` untuk Validasi

### Keputusan
Parameter `--limit-articles` tidak digunakan untuk mengambil sampel validasi.

### Alasan
Parameter tersebut mengurutkan artikel berdasarkan `title_score` dan mengambil artikel dengan skor tertinggi.

Cara tersebut dapat menyebabkan sampel validasi terlalu banyak berisi artikel yang memiliki keyword geopolitik pada judul, sehingga tidak merepresentasikan populasi artikel secara keseluruhan.

### Keputusan Sampling Validasi
Sampel validasi menggunakan **random sampling terstratifikasi berdasarkan tahun**.

---

## 10. Stratified Sampling Berdasarkan Tahun

### Keputusan
Validasi konten menggunakan **210 artikel**, yaitu:

- 35 artikel per tahun
- 6 tahun
- 2021–2026
- `random_state=42`

### Alasan
Distribusi artikel dapat berbeda antar tahun. Stratifikasi memastikan setiap tahun dalam rentang penelitian tetap terwakili.

### Hasil
Sampling terstratifikasi menghasilkan tingkat keberhasilan fetch sebesar **93,4%**.

---

## 11. Penanganan Artikel Berbayar

### Keputusan
Artikel CNBC Pro dan Investing Club dikeluarkan dari korpus.

### Alasan
Konten berbayar tidak dapat diperoleh secara konsisten melalui scraping biasa.

URL artikel berbayar juga tidak selalu dapat dibedakan secara langsung dari artikel reguler.

### Sinyal yang Digunakan
Metadata `section` digunakan dengan kondisi yang mendeteksi:

- `pro:`
- `investing club`

Sinyal tersebut digunakan sebagai indikator utama artikel berbayar.

---

## 12. Penanganan Live Blog

### Keputusan
Artikel **live blog tidak dibuang** dari korpus.

### Alasan
Live blog tetap dapat mengandung informasi geopolitik yang relevan.

Selain itu, pipeline preprocessing menyediakan `description` sebagai fallback ketika `body` tidak tersedia. Dengan demikian, artikel live blog masih dapat menyediakan informasi teks yang dapat diproses.

### Indikator Live Blog
Deteksi menggunakan indikator seperti:

- tanda `;` pada judul;
- frasa seperti `"live updates"`.

---

## 13. Penanganan Zona Waktu

### Keputusan
Konversi zona waktu hanya dilakukan pada timestamp berita CNBC.

`published_utc` dikonversi menjadi `published_wib` menggunakan zona waktu:

`Asia/Jakarta`

### Alasan
Timestamp CNBC tersedia dalam UTC, sedangkan JISDOR berkaitan dengan waktu operasional pasar Indonesia.

JISDOR tidak dikonversi dari UTC karena data tersebut merepresentasikan tanggal perdagangan BI, bukan timestamp berita yang perlu dikonversikan.

---

## 14. Penentuan Cutoff Berita terhadap JISDOR

### Keputusan
Menggunakan **08:00 WIB** sebagai cutoff pembentukan target.

### Aturan
- Berita sebelum **08:00 WIB** → dapat dikaitkan dengan fixing pada hari tersebut.
- Berita pada atau setelah **08:00 WIB** → digulirkan ke fixing hari kerja BI berikutnya.

### Alasan
JISDOR dihitung berdasarkan transaksi antarbank pada sekitar **08:00–09:45 WIB** dan dipublikasikan sekitar pukul 10:00 WIB.

Cutoff 08:00 dipilih agar berita yang muncul setelah awal jendela pembentukan fixing tidak dianggap sebagai informasi yang memengaruhi harga yang sudah mulai terbentuk.

### Alternatif yang Tidak Dipilih
Menggunakan pukul 10:00 WIB sebagai cutoff.

### Alasan Tidak Dipilih
Berita pada pukul 09:30 WIB, misalnya, memang terbit sebelum JISDOR dipublikasikan, tetapi informasi tersebut sudah muncul ketika transaksi yang membentuk fixing sedang berlangsung.

---

## 15. Penanganan Berita di Luar Hari Perdagangan

### Keputusan
Menggunakan **roll-forward** ke hari kerja BI berikutnya.

### Contoh
Berita yang diterbitkan pada Sabtu atau Minggu akan dikaitkan dengan fixing hari kerja berikutnya.

### Alasan
Informasi tidak seharusnya dikaitkan dengan fixing yang sudah terjadi pada masa lalu.

### Alternatif yang Dipertimbangkan

#### Roll-backward
Tidak digunakan karena dapat menyebabkan informasi masa depan dikaitkan dengan harga yang sudah terbentuk.

#### Membuang berita akhir pekan
Tidak digunakan karena dapat menghilangkan informasi geopolitik penting yang terjadi ketika pasar sedang tutup.

---

## 16. Penentuan Kalender Hari Kerja

### Keputusan
Kalender hari kerja diturunkan langsung dari data JISDOR.

### Alasan
JISDOR hanya memiliki data pada hari ketika fixing tersedia. Dengan demikian, hari tanpa entri JISDOR dapat digunakan untuk mengidentifikasi hari non-trading tanpa bergantung pada library kalender eksternal.

---

## 17. Mempertahankan Hari Tanpa Berita

### Keputusan
Hari kerja yang tidak memiliki berita geopolitik **tetap dipertahankan** dalam `aligned_daily.csv`.

### Representasi
- `n_news = 0`
- `text_norm_concat = ""`

### Alasan
Tidak adanya berita merupakan kondisi yang memiliki makna dalam deret waktu.

Menghapus hari tanpa berita dapat menyebabkan:
- kontinuitas deret waktu terganggu;
- observasi kurs hilang;
- model hanya melihat hari yang memiliki berita.

---

## 18. Normalisasi Volume Berita

### Keputusan
Tidak menggunakan `n_news` mentah sebagai satu-satunya ukuran volume berita.

Variabel tambahan:

`news_density = n_news / window_hours`

digunakan untuk mempertimbangkan perbedaan panjang jendela waktu.

### Alasan
Jendela berita untuk hari Senin lebih panjang karena mencakup periode akhir pekan.

Rata-rata:

- Senin: sekitar **62,85 jam**
- Hari kerja lainnya: sekitar **30 jam**

Namun jumlah berita tidak meningkat dua kali lipat. Hal ini menunjukkan bahwa jumlah berita mentah tidak dapat dibandingkan langsung antar hari tanpa memperhitungkan panjang jendela.

### Implikasi
`n_news / window_hours` dapat digunakan sebagai variabel kontrol dalam pemodelan atau analisis deskriptif.

---

## 19. Jalur Preprocessing TF-IDF dan BERT

### Keputusan
Membuat dua jalur preprocessing yang berbeda:

1. **TF-IDF**
2. **FinBERT/BERT**

### Alasan
Kedua jenis representasi teks memiliki kebutuhan preprocessing yang berbeda.

TF-IDF bekerja pada token yang telah dinormalisasi, sedangkan tokenizer BERT membutuhkan bentuk teks yang relatif alami agar proses subword tokenization tetap bekerja sebagaimana mestinya.

---

## 20. Preprocessing TF-IDF

### Keputusan
Pada jalur TF-IDF:

- melakukan lowercase;
- menghapus sebagian besar tanda baca;
- mempertahankan `.`, `$`, dan `%`;
- menggunakan custom stopword;
- mempertahankan kata negasi;
- tidak melakukan stemming;
- tidak melakukan lemmatization;
- melakukan deduplikasi;
- menghapus data yang bukan artikel valid.

### Alasan

#### Mempertahankan `$` dan `%`
Simbol tersebut dapat menunjukkan informasi ekonomi, misalnya:

- `oil surges 8%`
- `$100 billion in sanctions`

#### Mempertahankan `.`
Diperlukan untuk:
- angka desimal seperti `3.5`;
- singkatan seperti `U.S.`

#### Mempertahankan Negasi
Kata seperti:

- `not`
- `no`
- `never`
- `without`

dipertahankan karena penghapusannya dapat mengubah makna kalimat.

#### Tidak Melakukan Stemming
Stemming dapat menghasilkan bentuk kata yang tidak bermakna pada nama negara, organisasi, atau istilah geopolitik.

---

## 21. Preprocessing BERT

### Keputusan
Melakukan preprocessing seminimal mungkin pada teks yang akan diberikan kepada FinBERT.

### Keputusan
Tidak melakukan:

- lowercase manual;
- penghapusan tanda baca secara agresif;
- stopword removal;
- stemming;
- lemmatization.

### Alasan
Tokenizer BERT dirancang untuk memproses teks dalam bentuk yang relatif alami. Normalisasi agresif dapat mengubah representasi teks yang diharapkan oleh model pretrained.

---

## 22. Penghapusan Boilerplate pada BERT

### Keputusan
Menghapus boilerplate CNBC yang telah diuji sebelumnya, tetapi tidak melakukan penghapusan teks secara agresif.

### Contoh
Boilerplate yang dapat dihapus adalah bagian berulang seperti promosi atau footer tertentu.

### Alasan
Teks boilerplate tidak merepresentasikan informasi geopolitik dan dapat menjadi noise bagi model.

### Keputusan Tambahan
Pola boilerplate harus diuji terhadap keseluruhan korpus sebelum digunakan agar kalimat berita asli tidak ikut terhapus.

---

## 23. Penanganan Artikel Tanpa Body

### Keputusan
Jika body artikel tidak tersedia, menggunakan `description` sebagai fallback.

### Alasan
Sebagian artikel, terutama artikel dengan format live blog, tidak memiliki body yang dapat diekstraksi secara normal.

Sebanyak **454 artikel** menggunakan description sebagai fallback.

---

## 24. Penanganan Panjang Teks BERT

### Keputusan
Artikel yang melebihi kapasitas BERT diproses menggunakan **chunking 512 token dengan overlap 50 token**.

### Alasan
BERT memiliki batas panjang input sekitar 512 token.

Rata-rata artikel setelah preprocessing sekitar **710 token**, dan sekitar **73,4% artikel** melebihi batas tersebut.

### Alasan Menggunakan Overlap
Overlap 50 token digunakan untuk mengurangi risiko informasi penting terpotong tepat pada batas antar-chunk.

---

## 25. Pemrosesan Data Kurs JISDOR

### Keputusan
Data JISDOR dibersihkan dan diubah menjadi format kronologis dari tanggal terlama ke tanggal terbaru.

### Alasan
Data raw JISDOR tersusun dari tanggal terbaru ke tanggal terlama, sedangkan perhitungan perubahan nilai tukar membutuhkan informasi hari sebelumnya.

---

## 26. Parsing Format Tanggal JISDOR

### Keputusan
Format tanggal JISDOR dibaca secara eksplisit sebagai `M/D/YYYY` dan dikonversi menjadi `YYYY-MM-DD`.

### Alasan
Parsing eksplisit menghindari ambiguitas antara format bulan/hari dan hari/bulan.

Format standar `YYYY-MM-DD` juga memudahkan pengurutan berdasarkan waktu.

---

## 27. Parsing Nilai Kurs

### Keputusan
Parser kurs dibuat untuk menangani format angka Indonesia maupun Inggris.

### Contoh
- `15.250,00`
- `15,250.00`

### Alasan
Perbedaan pemisah ribuan dan desimal dapat menyebabkan nilai kurs terbaca secara salah apabila format tidak ditangani secara eksplisit.

---

## 28. Penghapusan Kolom NO JISDOR

### Keputusan
Kolom `NO` dihapus setelah data dibaca.

### Alasan
Kolom tersebut hanya merupakan nomor urut dan tidak memiliki informasi ekonomi yang digunakan dalam pemodelan.

---

## 29. Pembentukan Target Kurs

### Keputusan
Perubahan kurs dibandingkan dengan **hari kerja BI sebelumnya**, bukan tanggal kalender sebelumnya.

### Alasan
JISDOR tidak tersedia pada:
- Sabtu;
- Minggu;
- hari libur;
- cuti bersama.

Dengan menggunakan hari kerja BI sebelumnya, perubahan kurs Senin dibandingkan dengan Jumat sebelumnya, bukan Minggu.

---

## 30. Menyimpan Variabel Kontrol Alignment

### Keputusan
Menyimpan variabel kontrol berikut pada `aligned_daily.csv`:

- `is_offhours`
- `hours_to_fixing`
- `window_hours`
- `n_news`
- `target_date_lag1`

### Alasan
Variabel tersebut memungkinkan proses temporal alignment diperiksa dan digunakan dalam analisis robustness.

Keputusan ini mencegah efek dari perbedaan panjang jendela waktu tersembunyi di dalam dataset.

---

## 31. Mempertahankan Varian Lag-1

### Keputusan
`target_date_lag1` disimpan sebagai varian target untuk robustness check.

### Alasan
Variabel ini memungkinkan pengujian alternatif terhadap aturan alignment utama tanpa perlu mengulang seluruh proses akuisisi dan preprocessing.

---

## 32. Struktur Dataset Akhir

### Keputusan
Hasil temporal alignment disimpan dalam:

`aligned_daily.csv`

### Alasan
File ini menjadi titik pertemuan antara:

- representasi berita geopolitik;
- informasi tanggal;
- jumlah berita;
- informasi jendela waktu;
- nilai tukar;
- target perubahan kurs.

Dengan demikian, tahap pemodelan dapat dilakukan menggunakan dataset yang sudah terstruktur secara temporal.

---

## 33. Target Prediksi: Kelas Volatilitas

### Keputusan
Target model adalah `volatility_class` (`low` / `medium` / `high`), yaitu kelas dari `realized_vol`:

`realized_vol(t) = std(log_return[t-4], ..., log_return[t])`

atau simpangan baku perubahan kurs (log return) selama 5 hari kerja BI terakhir, termasuk hari t (`VOLATILITY_WINDOW = 5`).

### Alasan
- Pertanyaan penelitian berfokus pada **fluktuasi** nilai tukar, bukan arah naik/turun. Volatilitas lebih relevan untuk menangkap dampak ketidakpastian geopolitik.
- Jendela 5 hari kerja (±1 minggu) meredam derau harian satu hari.

### Konsekuensi yang Perlu Dipahami
Jendela hari t dan hari t-1 berbagi 4 dari 5 return yang sama, sehingga kelas volatilitas sangat persisten dari hari ke hari. Akibatnya baseline naive persistence (Keputusan 36) sudah kuat, dan fitur harga perlu dirancang mengikuti struktur ini (Keputusan 37).

### Alternatif yang Dipertimbangkan
- Arah perubahan kurs (`direction`: up/flat/down dengan `FLAT_THRESHOLD`).
- Volatilitas ke depan yang tidak tumpang tindih dengan data kemarin (mis. |r[t]| atau std r[t..t+4]).

### Alasan Tidak Dipilih
Kolom `direction` tetap dihitung di `preprocess_kurs.py`, tetapi tidak dipakai sebagai target karena pertanyaan penelitian adalah fluktuasi. Target volatilitas ke depan tidak dipilih untuk menjaga definisi target yang sudah ditetapkan; keterbatasan persistensinya dicatat sebagai bagian dari interpretasi hasil.

---

## 34. Ambang Kelas Volatilitas (Tertile dari Data Train)

### Keputusan
`realized_vol` dibagi menjadi 3 kelas menggunakan **tertile** (`pd.qcut(..., 3)`) yang dihitung **hanya dari 70% hari kerja tertua**. Ambang yang diperoleh:

- `low`: realized_vol < 0,207%
- `medium`: 0,207% – 0,312%
- `high`: > 0,312%

Ambang yang sama kemudian diterapkan (`pd.cut`) ke seluruh deret, termasuk valid dan test.

### Alasan
- **Kelas seimbang:** tertile membagi data train menjadi tiga kelompok berukuran sama, sehingga model tidak bisa mendapat akurasi tinggi hanya dengan menebak satu kelas.
- **Relatif terhadap perilaku rupiah:** "high" berarti lebih bergejolak dari kebiasaan USD/IDR selama periode train, bukan ambang absolut yang dipilih manual.
- **Mencegah kebocoran:** jika ambang dihitung dari seluruh data, ambang tersebut ikut "mengetahui" sebaran volatilitas periode test.

### Konsekuensi
Distribusi kelas di valid dan test tidak lagi seimbang (test: low 51, medium 65, high 51). Hal ini wajar karena kondisi pasar berubah antar periode.

---

## 35. Split Kronologis 70/15/15 pada Hari Berberita

### Keputusan
- Pelabelan kelas dan fitur lag dihitung pada **kalender hari kerja lengkap**, baru kemudian difilter ke hari yang memiliki berita (`n_news > 0`).
- Data dibagi secara kronologis: 70% train, 15% valid, 15% test.
- Ketiga skrip training (`train_baseline_ts.py`, `train_tfidf.py`, `train_combined.py`) memakai **himpunan hari yang identik**:
  - Train: 772 hari (16-09-2021 s.d. 25-03-2025)
  - Valid: 165 hari (26-03-2025 s.d. 09-12-2025)
  - Test: 167 hari (11-12-2025 s.d. 01-09-2026)

### Alasan
- Split acak akan membuat model belajar dari masa depan untuk memprediksi masa lalu.
- Valid dipakai untuk memilih hyperparameter; test hanya dipakai **satu kali** di akhir.
- Fitur lag dihitung sebelum filter agar "hari sebelumnya" selalu berarti hari kerja BI sebelumnya, bukan hari berberita sebelumnya.
- Himpunan hari yang identik diperlukan agar akurasi antar model dapat dibandingkan secara langsung.

---

## 36. Baseline Naive Persistence

### Keputusan
Baseline utama adalah **naive persistence**: kelas volatilitas hari t diprediksi sama dengan kelas hari t-1 (`lag1_volatility_class`). Baseline ini tidak dilatih dan tidak memiliki hyperparameter.

### Alasan
Karena target sangat persisten (Keputusan 33), naive persistence adalah tolok ukur minimum yang bermakna. Model yang tidak dapat melampaui naive tidak memberikan nilai tambah.

### Alternatif yang Tidak Dipilih
Dummy classifier (selalu menebak kelas mayoritas train). Baseline ini dihapus karena terlalu lemah (akurasi test 0,305, setara menebak acak) dan tidak relevan untuk pertanyaan penelitian.

---

## 37. Fitur Harga (`PRICE_FEATURES`)

### Keputusan
Fitur harga didefinisikan satu kali di `dataset_split.py` dan dipakai **sama persis** oleh XGBoost tanpa NLP dan XGBoost + TF-IDF + LM:

| Fitur | Isi |
|---|---|
| `lag1_log_return` s.d. `lag4_log_return` | Return 1–4 hari kerja sebelumnya |
| `known4_mean`, `known4_std` | Rata-rata dan simpangan baku dari r[t-4..t-1] |
| `vol_if_r0` | `realized_vol(t)` jika r[t] = 0 |
| `vol_if_rtyp_pos`, `vol_if_rtyp_neg` | `realized_vol(t)` jika r[t] = ± median \|r\| historis (expanding, di-shift 1 hari) |
| `lag1_volatility_class_enc` | Kelas kemarin (low = 0, medium = 1, high = 2) |

### Alasan
Empat dari lima return dalam jendela target sudah diketahui sebelum hari t (Keputusan 33). Fitur di atas memberikan informasi tersebut secara langsung, sehingga model cukup menilai apakah pergerakan hari t (dan informasi berita) cukup besar untuk memindahkan kelas.

Semua fitur hanya menggunakan data **sebelum hari t**. Hal ini telah diuji: mengubah return hari t dan sesudahnya tidak mengubah nilai fitur hari t.

### Alternatif yang Dipertimbangkan
Fitur awal: `lag1–3_log_return`, `lag1_realized_vol`, `lag1_volatility_class_enc`.

### Alasan Tidak Dipilih
- `lag4_log_return` tidak tersedia, padahal r[t-4] masih termasuk dalam jendela target.
- `lag1_realized_vol` memuat r[t-5], yaitu return yang sudah keluar dari jendela target.
- Dengan fitur awal, XGBoost tanpa NLP hanya mencapai akurasi test 0,689, lebih rendah dari naive (0,713).

---

## 38. Tipe Numerik untuk Fitur Kelas Kemarin

### Keputusan
`lag1_volatility_class_enc` disimpan sebagai **float** (`.astype(float)`).

### Alasan
Tanpa konversi, kolom tersebut mewarisi tipe `category` dari `pd.cut`. XGBoost membacanya sebagai kategori tanpa urutan jika diberi DataFrame (XGBoost tanpa NLP dan ablasi), tetapi sebagai angka berurutan jika diberi array numpy (model gabungan). Akibatnya fitur yang sama diperlakukan berbeda antar model yang dibandingkan. Pada versi XGBoost lama, kondisi ini juga dapat menimbulkan error.

---

## 39. Normalisasi Fitur Sentimen LM (Rata-rata per Artikel)

### Keputusan
Fitur sentimen Loughran-McDonald yang dipakai model:

- `mean_lm_polarity`
- `lm_positive_mean = lm_positive_sum / n_news`
- `lm_negative_mean = lm_negative_sum / n_news`

Fitur ini didefinisikan sekali sebagai `LM_FEATURES` di `dataset_split.py`.

### Alasan
- **Konsisten dengan Keputusan 18.** Jumlah berita mentah tidak dapat dibandingkan langsung antar hari. Fitur berbasis jumlah (`lm_*_sum`) memiliki masalah yang sama karena nilainya ikut membesar seiring jumlah artikel.
- **Fitur sentimen seharusnya mengukur nada, bukan volume.** Volume berita sudah terwakili terpisah melalui `n_news`.
- **Temuan EDA:** jumlah artikel per hari tidak stabil antar periode (rata-rata 3,9 pada train dan 8,0 pada valid), diduga akibat perbedaan cakupan scraping. Sementara itu, jumlah kata negatif per artikel relatif stabil (±21–27). Pergeseran ini sudah terlihat pada train dan valid, tanpa perlu melihat data test.

### Alternatif yang Dipertimbangkan
`lm_positive_sum` dan `lm_negative_sum` (jumlah per hari).

### Catatan Transparansi
Kedua versi telah dibandingkan pada model utama. Versi rata-rata: akurasi valid 0,812 dan test 0,760. Versi jumlah: akurasi valid 0,806 dan test 0,766. Selisih di test hanya satu hari dan justru sedikit berpihak pada versi jumlah, sehingga keputusan ini tidak mengubah kesimpulan dan tidak didasarkan pada hasil test; versi rata-rata dipilih karena alasan di atas (dan akurasi valid-nya juga lebih tinggi).

---

## 40. Representasi TF-IDF pada Model Gabungan

### Keputusan
- TF-IDF: `min_df=5`, `max_df=0.8`, unigram + bigram, maksimum 5.000 fitur, di-fit dari data train saja.
- Untuk model gabungan, TF-IDF dikompres menjadi 20 komponen dengan `TruncatedSVD` yang juga di-fit dari data train saja.

### Alasan
Tanpa kompresi, 5.000 kolom TF-IDF akan mendominasi pemilihan split pada pohon XGBoost, sehingga fitur harga dan LM yang jumlahnya jauh lebih sedikit hampir tidak pernah terpakai. Model TF-IDF saja (`train_tfidf.py`) tetap memakai 5.000 kolom penuh karena tidak digabung dengan fitur lain.

---

## 41. XGBoost dengan Tuning Hyperparameter yang Seragam

### Keputusan
Semua model XGBoost di-tuning menggunakan fungsi dan grid yang sama (`tune_xgb` dan `XGB_GRID` di `dataset_split.py`):

- `n_estimators`: 100, 200, 400
- `max_depth`: 2, 3, 4
- `learning_rate`: 0,03; 0,05; 0,1

Total 27 kombinasi. Setiap kombinasi dilatih pada train dan dinilai dengan akurasi valid. Kombinasi terbaik dievaluasi pada test **satu kali**. Parameter lain memakai default XGBoost dengan `random_state=42`.

### Alasan
- Sebelumnya hanya model gabungan yang di-tuning, sedangkan model pembanding memakai parameter tetap. Perbandingan seperti itu tidak adil karena model NLP mendapat kesempatan optimasi lebih banyak.
- Dengan grid yang sama, selisih akurasi antar model mencerminkan kontribusi fitur, bukan perbedaan upaya tuning.

### Alternatif yang Tidak Dipilih
Logistic Regression sebagai algoritma pembanding pada model teks saja. Algoritma ini dihapus karena perbandingan antar algoritma bukan bagian dari pertanyaan penelitian.

---

## 42. Model yang Dibandingkan

### Keputusan
**Perbandingan utama** (`train_combined.py`):

1. Naive persistence (baseline)
2. XGBoost tanpa NLP (`PRICE_FEATURES`)
3. XGBoost + TF-IDF + LM (model utama)

**Pendukung:**

- XGBoost teks saja: TF-IDF saja dan LM saja, tanpa fitur harga (`train_tfidf.py`).
- Ablasi: XGBoost harga + LM tanpa TF-IDF (`train_combined.py`).

### Alasan
- Selisih (3) terhadap (2) mengukur kontribusi berita geopolitik di atas histori harga, yang merupakan inti pertanyaan penelitian.
- Model teks saja menguji apakah berita dapat memprediksi volatilitas **tanpa** histori harga.
- Ablasi memisahkan kontribusi LM dari TF-IDF.

---

## 43. Metrik Evaluasi dan Uji Signifikansi

### Keputusan
- Metrik utama: **akurasi** pada test set. Metrik pendukung: macro F1, classification report, dan confusion matrix.
- Signifikansi selisih antar model diuji dengan **uji McNemar exact** (α = 0,05) pada test set. Uji ini sudah dijalankan otomatis di `train_combined.py`.

### Alasan
- Kelas pada data train seimbang, sehingga akurasi mudah diinterpretasikan. Macro F1 ditambahkan karena distribusi kelas test tidak seimbang.
- Uji McNemar sesuai untuk membandingkan dua classifier pada data uji yang sama: yang dihitung hanya hari-hari ketika kedua model memberikan hasil berbeda.

---

## 44. Hasil Evaluasi Akhir dan Keterbatasan

### Hasil (test set, 167 hari)

| Model | Akurasi valid | Akurasi test | Macro F1 test |
|---|---|---|---|
| Naive persistence | 0,727 | 0,713 | 0,72 |
| XGBoost teks saja (TF-IDF) | 0,382 | 0,287 | 0,21 |
| XGBoost teks saja (LM) | 0,370 | 0,305 | 0,31 |
| XGBoost tanpa NLP | 0,806 | 0,731 | 0,74 |
| Ablasi: harga + LM | 0,818 | 0,737 | – |
| **XGBoost + TF-IDF + LM** | **0,812** | **0,760** | **0,77** |

Uji McNemar (hari "A saja benar" / "B saja benar"):

- XGBoost tanpa NLP vs naive: 20 / 17, p = 0,743
- XGBoost + TF-IDF + LM vs XGBoost tanpa NLP: 8 / 3, p = 0,227
- XGBoost + TF-IDF + LM vs naive: 23 / 15, p = 0,256

### Interpretasi
- Histori harga merupakan sumber informasi utama. Total feature importance model utama: harga 0,69, TF-IDF 0,26, LM 0,05.
- Berita geopolitik menambah akurasi sebesar +3,0 poin di atas model harga saja, terutama pada kelas `medium`. Namun peningkatan ini **tidak signifikan secara statistik** pada α = 0,05.
- Berita **tidak mampu memprediksi sendirian**. Model teks saja berada pada tingkat tebakan acak, dan model TF-IDF saja cenderung menghafal kosakata periode tertentu (misalnya "invasion", "annexation crimea") alih-alih pola yang dapat digeneralisasi.
- TF-IDF (topik berita) lebih informatif dibanding nada sentimen LM.

### Keterbatasan
- Test set kecil (167 hari): satu hari setara 0,6 poin akurasi.
- Evaluasi hanya pada satu periode test.
- Jumlah artikel per hari meningkat pada 2025–2026, kemungkinan akibat cakupan scraping.
- Fitur harga (Keputusan 37) dirancang pada tahap eksplorasi ketika hasil test versi awal sudah terlihat. Rancangannya diturunkan dari definisi target, bukan dari coba-coba, tetapi hal ini tetap dicatat sebagai keterbatasan.

---

## 45. Pemeriksaan Pergeseran Distribusi Fitur Antar Periode

### Keputusan

Memasukkan pemeriksaan distribusi fitur antar periode sebagai bagian dari **Exploratory Data Analysis (EDA)**. Perbandingan train dan valid digunakan untuk mengidentifikasi pergeseran sebelum evaluasi akhir pada test.

### Bukti dari EDA

Berdasarkan ringkasan EDA, rata-rata jumlah artikel per hari berubah dari sekitar **3,9 pada train** menjadi **8,0 pada valid**. Perbedaan ini menunjukkan perubahan rata-rata volume berita antar periode, sehingga indikasi pergeseran sudah dapat diketahui tanpa melihat test.

Perubahan cakupan scraping merupakan salah satu kemungkinan penyebabnya. Namun, penyebab tersebut belum dapat dipastikan hanya dari rata-rata jumlah artikel; perubahan intensitas pemberitaan dan komposisi periode juga perlu dipertimbangkan.

### Cakupan Pemeriksaan

- Jumlah artikel per tahun dan per hari untuk menggambarkan cakupan hasil scraping dari waktu ke waktu. Perbandingan antar tahun perlu memperhatikan apakah tahun tersebut tercakup penuh atau hanya sebagian.
- Distribusi `n_news` antar periode, termasuk median, kuartil, dan proporsi hari tanpa berita, agar analisis tidak hanya bergantung pada rata-rata.
- Sebaran fitur teks, misalnya `lm_negative_sum`, pada train dan valid untuk memeriksa apakah perubahan volume berita turut memengaruhi fitur agregat.

Pemeriksaan ini termasuk **distribution shift pada fitur**. Istilah *covariate shift* secara lebih khusus mengasumsikan bahwa distribusi fitur berubah sementara hubungan bersyarat antara fitur dan target tetap; pemeriksaan sebaran fitur saja belum membuktikan asumsi tersebut.

### Implikasi

Fitur berbentuk jumlah, seperti `lm_negative_sum`, dapat meningkat karena lebih banyak artikel atau teks yang terkumpul. Karena itu, kenaikannya tidak langsung diartikan sebagai peningkatan intensitas sentimen negatif. Analisis perlu mempertimbangkan volume berita dan panjang teks, serta membandingkan fitur agregat dengan ukuran yang dinormalisasi bila sesuai.

Normalisasi `news_density` pada keputusan 18 memperhitungkan panjang jendela waktu, tetapi tidak otomatis mengatasi perubahan cakupan scraping antar periode.

### Batas Penggunaan Data

Test disimpan untuk evaluasi akhir dan tidak digunakan untuk menentukan fitur, normalisasi, atau parameter model. Transformasi yang mempelajari parameter dari data, seperti scaler dan TF-IDF, di-fit hanya pada train, kemudian diterapkan pada valid dan test.


---

## 46. Tujuan dan Cakupan Early Fusion

### Keputusan
Early fusion versi penuh (`train_early_fusion.py`) menggabungkan seluruh blok fitur menjadi **satu vektor per hari** yang dilatih dengan **satu classifier**. Pertanyaan yang dijawab:

> Apakah embedding kontekstual, topik, dan NER **menambah informasi di atas** model utama Tugas 2 (harga + TF-IDF + LM)?

Fitur harga (`PRICE_FEATURES`) selalu disertakan. Pembanding utamanya adalah XGBoost tanpa NLP (F0) dan XGBoost + TF-IDF + LM (F1).

### Alasan
- Keputusan 44 sudah menunjukkan bahwa teks saja tidak mampu memprediksi kelas volatilitas (setara tebakan acak). Mengulang eksperimen teks saja dengan fitur yang lebih banyak tidak menjawab pertanyaan baru.
- Inti pertanyaan penelitian (Keputusan 42) adalah kontribusi berita **di atas histori harga**. Early fusion tanpa fitur harga tidak dapat dibandingkan dengan model utama.
- `evaluate_feature_fusion.py` (E1–E4) hanya menggabungkan embedding + topik + NER dengan Logistic Regression dan tanpa fitur harga, sehingga hasilnya tidak sebanding dengan Keputusan 44. Skrip tersebut dipertahankan sebagai eksperimen pendukung, bukan sebagai early fusion final.

### Alternatif yang Tidak Dipilih
Late fusion (menggabungkan prediksi beberapa model) dan hybrid fusion. Keduanya berada di luar cakupan tugas ini.

---

## 47. Embedding yang Digunakan dan Cara Pemakaiannya

### Keputusan
Embedding yang dipakai adalah tiga model kontekstual, masing-masing dalam dua jenis teks (6 varian):

| Model | Checkpoint | Dimensi | Jenis teks |
|---|---|---|---|
| FinBERT | `ProsusAI/finbert` | 768 | headline, fulltext |
| RoBERTa | `FacebookAI/roberta-base` | 768 | headline, fulltext |
| Qwen | `Qwen/Qwen2.5-0.5B` | 896 | headline, fulltext |

Cara pemakaian:

1. **Enam run terpisah** (eksperimen utama): setiap run memakai satu varian embedding bersama blok lain.
2. **Satu run "kitchen sink"**: keenam varian digabung sekaligus, masing-masing direduksi terlebih dahulu (Keputusan 48).

### Alasan
- Run terpisah memungkinkan perbandingan langsung antar model dan antara headline dengan fulltext.
- Run gabungan menguji apakah menumpuk seluruh representasi memberikan manfaat tambahan.
- Embedding statis (Word2Vec, FastText, Doc2Vec) tidak disertakan agar jumlah kombinasi tetap terkendali; ketiganya sudah dievaluasi di modul `head_vs_full`.

### Alternatif yang Tidak Dipilih
Menggabungkan headline dan fulltext per model (3 run). Cara ini menghilangkan perbandingan headline vs fulltext.

---

## 48. Reduksi Dimensi Embedding

### Keputusan
Setiap embedding harian diproses dengan `StandardScaler` lalu `PCA` menjadi **20 komponen**. Scaler dan PCA di-fit **hanya pada data train**. Pada run gabungan, PCA dilakukan **per embedding** (6 × 20 kolom), bukan setelah digabung.

### Alasan
- Alasan yang sama dengan Keputusan 40: 768–896 kolom embedding akan mendominasi pemilihan split pada pohon XGBoost, sehingga 10 fitur harga hampir tidak terpakai. Data train juga hanya 772 hari, lebih sedikit dari jumlah kolom embedding.
- Jumlah komponen disamakan dengan TF-IDF (20) agar setiap blok teks memiliki jumlah kolom yang setara.
- Standardisasi diperlukan karena skala Qwen berbeda dengan FinBERT/RoBERTa; tanpa itu PCA didominasi beberapa dimensi berskala besar.
- PCA dipilih (bukan TruncatedSVD) karena embedding bersifat dense dan perlu dipusatkan.

### Alternatif yang Tidak Dipilih
- Memilih jumlah komponen (10/20/50) lewat valid: melipatgandakan jumlah run dan peluang overfitting ke valid.
- Ambang explained variance: jumlah komponen menjadi berbeda antar model sehingga perbandingan tidak setara.
- UMAP: tidak deterministik dan rawan overfitting pada data kecil.

---

## 49. Agregasi Temporal Embedding

### Keputusan
- Agregasi default untuk semua embedding: **A0 (rata-rata harian)**.
- Satu agregasi alternatif, **A2 exponential decay dengan half-life 3 hari**, dijalankan sebagai ablasi hanya pada embedding model final.

### Alasan
- Ranking `temporal_aggregation` menunjukkan pemilihan agregasi berdasarkan valid tidak stabil: konfigurasi terbaik di valid (akurasi 0,44) hanya mencapai 0,29 di test.
- Memilih agregasi per embedding berarti 6 × 12 = 72 kombinasi, yang memperbesar peluang menemukan konfigurasi "terbaik" secara kebetulan.
- A2 half-life 3 hari dipilih atas dasar teori (pengaruh berita terhadap kurs memudar dalam beberapa hari), bukan dari skor. Jendelanya hanya memakai hari ≤ t.

---

## 50. Blok Dasar pada Early Fusion

### Keputusan
Dua susunan dasar dijalankan untuk setiap embedding:

- **Dasar B (utama):** harga + LM + TF-IDF (= F1, model utama Tugas 2), lalu ditambah embedding, topik, dan NER.
- **Dasar A (pendukung):** harga + LM + embedding, tanpa TF-IDF.

### Alasan
- Dasar B menjawab apakah fitur baru menambah informasi di atas model terbaik yang sudah ada.
- Dasar A membandingkan embedding dengan TF-IDF secara langsung sebagai representasi teks, karena keduanya menempati posisi yang sama dengan jumlah kolom yang sama (20).

---

## 51. Fitur Topik: LDA yang Di-fit pada Train Saja

### Keputusan
- Model topik: **LDA 15 topik**, dengan tokenizer dan hyperparameter yang sama dengan `topic_models.py`.
- LDA (termasuk `CountVectorizer`) **di-fit ulang hanya pada artikel dengan `target_date` ≤ akhir periode train**, kemudian diterapkan (`transform`) ke seluruh artikel.
- Fitur harian: rata-rata distribusi topik per hari (`topic_XX_mean`, 15 kolom). Kolom `topic_XX_count` tidak dipakai.

### Alasan
- Fitur topik sebelumnya berasal dari model yang di-fit pada seluruh rentang waktu, sehingga bersifat transduktif dan melanggar batas penggunaan data pada Keputusan 45. Catatan ini sudah tercantum di metadata `feature_fusion_evaluation`.
- `selected_model.json` menunjuk Top2Vec, tetapi statusnya belum final dan nilai coherence-nya kosong (tidak ada topik yang dapat dievaluasi), sehingga skornya tidak dapat diandalkan. BERTopic memiliki 27% dokumen outlier. LDA memiliki coherence yang valid, mencakup seluruh dokumen, dan berdimensi kecil.
- Kolom jumlah (`_count`) membesar seiring volume berita (masalah yang sama dengan Keputusan 39).

---

## 52. Fitur NER: Normalisasi per Artikel dan Pemilihan Entitas

### Keputusan
Seluruh fitur NER dibagi dengan `n_news` sehingga menjadi ukuran **per artikel**. Fitur yang dipakai (21 kolom):

- **Tingkat grup (7):** rata-rata entitas per artikel; proporsi artikel yang menyebut negara, bank sentral, dan peristiwa ekonomi; rata-rata penyebutan ketiga grup tersebut per artikel.
- **Entitas kunci (14):** United States, China, Russia, Ukraine, Israel, Iran, Indonesia; Federal Reserve, Bank Indonesia; peristiwa kebijakan moneter, sanksi, kebijakan dagang, nilai tukar, dan energi/komoditas.

### Alasan
- Kolom mentah `*_mentions`, `*_count`, `has_*`, dan `n_unique_*` adalah **jumlah harian**, sehingga ikut membesar ketika jumlah artikel per hari naik (rata-rata 3,9 pada train dan 8,0 pada valid). Ini sejalan dengan Keputusan 18 dan 39.
- Sebagian besar dari ±50 entitas spesifik bernilai nol pada kebanyakan hari, sehingga hanya menambah dimensi.
- Daftar entitas kunci ditetapkan dari pertimbangan domain (pelaku utama yang relevan bagi USD/IDR dan konflik utama periode penelitian), **bukan** dari hasil valid atau test.

### Alternatif yang Tidak Dipilih
Memakai seluruh entitas, atau menyaring entitas berdasarkan frekuensi.

---

## 53. Classifier dan Tuning pada Early Fusion

### Keputusan
Semua run memakai **XGBoost dengan `tune_xgb` dan `XGB_GRID`** yang sama dengan Keputusan 41 (27 kombinasi, dipilih berdasarkan akurasi valid, `random_state=42`).

### Alasan
Dengan algoritma dan upaya tuning yang seragam, selisih akurasi antar run mencerminkan kontribusi fitur, dan hasilnya dapat dibandingkan langsung dengan Keputusan 44.

### Alternatif yang Tidak Dipilih
Logistic Regression (dipakai di `evaluate_feature_fusion.py`). Perbandingan antar algoritma bukan bagian dari pertanyaan penelitian (Keputusan 41).

---

## 54. Fitur Turunan Berdimensi Kecil (Ablasi)

### Keputusan
Tiga fitur turunan diuji sebagai **ablasi** di atas model final, bukan sebagai bagian dari model final:

- `news_novelty`: 1 − cosine antara embedding hari t dan rata-rata embedding 30 hari berberita sebelumnya.
- `topic_entropy`: entropi distribusi topik harian.
- `topic_shift`: jarak Jensen-Shannon antara distribusi topik hari t dan rata-rata 7 hari berberita sebelumnya.

Ketiganya hanya memakai data hari ≤ t.

### Ditunda
- Probabilitas sentimen FinBERT (positive/negative/neutral): memerlukan inferensi ulang karena yang tersimpan hanya embedding.
- Dispersi embedding dalam satu hari: memerlukan embedding tingkat artikel, sedangkan yang tersimpan sudah diagregasi harian.

---

## 55. Desain Eksperimen dan Aturan Pemilihan Model Final

### Keputusan
Susunan run:

| Kelompok | Isi |
|---|---|
| Baseline | Naive persistence; F0 = harga; F1 = harga + LM + TF-IDF |
| Tanpa embedding | F1 + topik; F1 + NER; F1 + topik + NER |
| Per embedding (×6) | harga + LM + emb; F1 + emb; F1 + emb + topik; F1 + emb + NER; **FULL = F1 + emb + topik + NER** |
| Kitchen sink | F1 + 6 embedding + topik + NER |
| Ablasi model final | leave-one-block-out; agregasi A2; fitur turunan |

**Model final** adalah run FULL dengan **akurasi valid tertinggi** di antara enam embedding (tie-break: macro F1 valid, lalu urutan tetap FinBERT → RoBERTa → Qwen, headline → fulltext). Aturan ini ditetapkan sebelum eksperimen dijalankan.

### Alasan
- Hanya model final yang diklaim pada test. Run lain dilaporkan sebagai analisis, karena dengan puluhan run selalu ada peluang salah satunya unggul di test secara kebetulan.
- Leave-one-block-out mengukur kontribusi setiap blok terhadap model final.
- Seluruh run memakai himpunan hari yang identik dengan Keputusan 35 (772 / 165 / 167); skrip menghentikan proses jika jumlahnya berbeda.

---

## 56. Signifikansi dan Ketidakpastian pada Early Fusion

### Keputusan
- **Uji McNemar exact** (Keputusan 43) untuk model final terhadap F1, F0, dan naive persistence, dengan **koreksi Holm** atas ketiga perbandingan.
- **Interval kepercayaan 95% bootstrap** (5.000 resample hari test) untuk akurasi dan untuk selisih akurasi berpasangan.
- **Multi-seed** (5 seed) pada hyperparameter terpilih.

### Alasan
- Test hanya 167 hari (satu hari setara 0,6 poin akurasi), sehingga selisih kecil perlu disertai ukuran ketidakpastian.
- Koreksi Holm diperlukan karena lebih dari satu perbandingan diuji sekaligus.

### Catatan
XGBoost tanpa subsampling bersifat deterministik, sehingga pergantian seed tidak mengubah hasil. Multi-seed tetap dijalankan sebagai pemeriksaan, tetapi ketidakpastian dilaporkan melalui interval bootstrap.

---

## 57. Hasil Evaluasi Early Fusion dan Interpretasi

Sumber angka: `logs/train_early_fusion_report.txt` dan `data/processed/early_fusion/` (41 run XGBoost, split 772 / 165 / 167 hari). Hanya **model final** yang diklaim pada test; seluruh run lain dilaporkan sebagai analisis (Keputusan 55).

### Model Final
Run FULL dengan akurasi valid tertinggi adalah **`FULL_roberta_fulltext`** (harga + LM + TF-IDF + RoBERTa fulltext + topik + NER, 89 fitur), tanpa seri sehingga tie-break tidak terpakai. Hyperparameter terpilih: `n_estimators=100`, `max_depth=4`, `learning_rate=0,05`.

### Hasil Utama (test set, 167 hari)

| Model | Fitur | Akurasi valid | Akurasi test | CI 95% bootstrap (test) | Macro F1 test |
|---|---|---|---|---|---|
| Naive persistence | 0 | 0,727 | 0,713 | [0,647; 0,778] | 0,715 |
| F0: XGBoost tanpa NLP (harga) | 10 | 0,806 | 0,731 | [0,659; 0,796] | 0,737 |
| F1: harga + LM + TF-IDF (model utama Tugas 2) | 33 | 0,812 | 0,760 | [0,695; 0,820] | 0,766 |
| **FINAL: `FULL_roberta_fulltext`** | **89** | **0,812** | **0,766** | **[0,701; 0,826]** | **0,771** |

Dalam jumlah hari yang ditebak benar: naive 119, F0 122, F1 127, FINAL 128 dari 167. Selisih FINAL terhadap F1 adalah **satu hari**.

F0 dan F1 mereproduksi `logs/train_combined_report.txt` secara persis (akurasi valid dan test, hyperparameter, total importance per blok, dan confusion matrix F1 identik).

Confusion matrix FINAL pada test (baris = aktual, kolom = prediksi; urutan high, low, medium):

| | high | low | medium |
|---|---|---|---|
| **high** | 36 | 1 | 14 |
| **low** | 0 | 42 | 9 |
| **medium** | 6 | 9 | 50 |

Recall per kelas: high 0,71; low 0,82; medium 0,77. Kesalahan hampir seluruhnya terjadi antara kelas bertetangga (`medium` dengan `high` atau `low`).

### Uji Signifikansi dan Ketidakpastian

Uji McNemar exact (hari "FINAL saja benar" / "pembanding saja benar"), dengan koreksi Holm atas tiga perbandingan:

| Perbandingan | Hari | p mentah | p Holm | CI 95% bootstrap selisih akurasi | Kesimpulan |
|---|---|---|---|---|---|
| FINAL vs F1 | 6 / 5 | 1,000 | 1,000 | [−0,036; +0,048] | tidak signifikan |
| FINAL vs F0 | 12 / 6 | 0,238 | 0,633 | [−0,012; +0,084] | tidak signifikan |
| FINAL vs naive | 25 / 16 | 0,211 | 0,633 | [−0,024; +0,132] | tidak signifikan |

Ketiga interval selisih memuat nol. Multi-seed (5 seed) menghasilkan akurasi test yang sama persis (F1 0,760 ± 0,000; FINAL 0,766 ± 0,000), sesuai catatan Keputusan 56: XGBoost tanpa subsampling deterministik, sehingga ketidakpastian dibaca dari interval bootstrap, bukan dari seed.

### Enam Run FULL dan Kitchen Sink (analisis)

| Run | Fitur | Akurasi valid | Akurasi test | Macro F1 test |
|---|---|---|---|---|
| FULL FinBERT headline | 89 | 0,788 | 0,754 | 0,760 |
| FULL FinBERT fulltext | 89 | 0,806 | 0,760 | 0,767 |
| FULL RoBERTa headline | 89 | 0,794 | 0,754 | 0,760 |
| **FULL RoBERTa fulltext (FINAL)** | 89 | **0,812** | 0,766 | 0,771 |
| FULL Qwen headline | 89 | 0,788 | 0,754 | 0,760 |
| FULL Qwen fulltext | 89 | 0,806 | 0,754 | 0,761 |
| Kitchen sink (6 embedding) | 189 | 0,800 | 0,737 | 0,744 |

- Keenam run FULL berada pada rentang test 0,754–0,766, yaitu 126–128 hari benar. Perbedaan antar model embedding dan antara headline dengan fulltext hanya dua hari, sehingga **tidak ada model embedding yang dapat dinyatakan lebih baik**.
- Varian fulltext memiliki akurasi valid lebih tinggi daripada headline pada ketiga model (0,806–0,812 vs 0,788–0,794), tetapi pola ini tidak terbawa ke test.
- Kitchen sink lebih rendah daripada F1 (0,737 vs 0,760) walaupun memakai fitur hampir enam kali lebih banyak. Menumpuk seluruh representasi tidak memberi manfaat.

### Embedding sebagai Pengganti TF-IDF (analisis)

Dasar A (harga + LM + embedding, 33 fitur) dibandingkan dengan F1 (harga + LM + TF-IDF, 33 fitur; valid 0,812, test 0,760):

| Embedding | Akurasi valid | Akurasi test | Macro F1 test |
|---|---|---|---|
| FinBERT headline | 0,800 | 0,766 | 0,772 |
| FinBERT fulltext | 0,818 | 0,737 | 0,744 |
| RoBERTa headline | 0,806 | 0,749 | 0,756 |
| RoBERTa fulltext | 0,818 | 0,778 | 0,783 |
| Qwen headline | 0,782 | 0,754 | 0,761 |
| Qwen fulltext | 0,812 | 0,725 | 0,731 |

- Dua dari enam embedding berada di atas F1 pada test dan empat di bawahnya (rentang 0,725–0,778, atau 121–130 hari benar dibanding 127 hari untuk F1). **Embedding kontekstual tidak secara konsisten lebih baik daripada TF-IDF.**
- Angka tertinggi (RoBERTa fulltext, 0,778) hanya unggul tiga hari dari F1. Dihitung dari `early_fusion_test_predictions.csv`, perbandingannya 8 / 5 hari dengan p McNemar = 0,581. Uji ini bersifat eksploratif, tidak dikoreksi, dan run ini bukan model final, sehingga tidak diklaim.
- Menambahkan embedding **di atas** F1 (F1 + embedding, 53 fitur) menghasilkan test 0,743–0,754 untuk keenam embedding, semuanya di bawah F1.

### Blok Topik dan NER di Atas F1 (analisis)

| Run | Fitur | Akurasi valid | Akurasi test |
|---|---|---|---|
| F1 + topik | 48 | 0,824 | 0,760 |
| F1 + NER | 54 | 0,794 | 0,749 |
| F1 + topik + NER | 69 | 0,794 | 0,749 |

Topik menaikkan akurasi valid tetapi akurasi test sama dengan F1. NER menurunkan keduanya.

### Feature Importance per Blok

| Model | Harga | LM | TF-IDF | Embedding | Topik | NER |
|---|---|---|---|---|---|---|
| F1 | 0,690 | 0,054 | 0,257 | – | – | – |
| FINAL | 0,400 | 0,030 | 0,167 | 0,170 | 0,134 | 0,099 |
| Kitchen sink | 0,346 | 0,028 | 0,053 | 0,455 | 0,050 | 0,067 |

Harga tetap menjadi blok terbesar. Namun porsi importance blok teks membesar seiring jumlah kolomnya (dari 23 kolom teks pada F1 menjadi 79 pada FINAL dan 179 pada kitchen sink), sedangkan akurasi test tidak ikut naik. Total importance per blok menunjukkan seberapa sering dan seberapa besar sebuah blok dipakai pohon, **bukan** bukti bahwa blok tersebut menambah kemampuan prediksi.

### Ablasi Model Final (analisis)

Leave-one-block-out (FINAL: valid 0,812, test 0,766):

| Blok dibuang | Akurasi valid | Akurasi test | Selisih test vs FINAL |
|---|---|---|---|
| Harga | 0,358 | 0,329 | −0,437 |
| LM | 0,818 | 0,784 | +0,018 |
| TF-IDF | 0,818 | 0,784 | +0,018 |
| Embedding | 0,794 | 0,749 | −0,018 |
| Topik | 0,830 | 0,760 | −0,006 |
| NER | 0,818 | 0,749 | −0,018 |

- Hanya blok harga yang kontribusinya jelas: tanpa harga, akurasi jatuh ke 0,329 (macro F1 0,254), setara tebakan acak. Ini konsisten dengan Keputusan 44 bahwa teks tidak mampu memprediksi sendirian, dan berlaku juga setelah embedding, topik, dan NER ditambahkan.
- Untuk lima blok teks, selisihnya hanya −3 sampai +3 hari dan arahnya tidak seragam (membuang LM atau TF-IDF justru menaikkan test). Seluruhnya berada di dalam lebar interval bootstrap akurasi FINAL (±6 poin), sehingga **tidak ada blok teks yang kontribusinya dapat dibedakan dari derau**.
- Angka 0,784 tidak diklaim: kedua run tersebut bukan kandidat model final menurut aturan Keputusan 55.

Ablasi lain:

| Run | Fitur | Akurasi valid | Akurasi test | Macro F1 test |
|---|---|---|---|---|
| FINAL dengan agregasi A2 (exp. decay, half-life 3 hari) | 89 | 0,794 | 0,725 | 0,730 |
| FINAL + fitur turunan (novelty, entropy, shift) | 92 | 0,806 | 0,743 | 0,749 |

- **Agregasi A2** lebih rendah daripada A0 di valid maupun test (0,725 vs 0,766 pada test), sehingga A0 dipertahankan sebagai default (Keputusan 49).
- **Fitur turunan** tidak membantu: valid dan test turun, dan total importance ketiganya hanya 0,033. Fitur ini tetap berstatus ablasi (Keputusan 54).

### Interpretasi
- **Jawaban atas pertanyaan Keputusan 46: tidak ada bukti bahwa embedding kontekstual, topik, dan NER menambah informasi di atas model utama Tugas 2.** FINAL hanya unggul satu hari dari F1 (0,766 vs 0,760; McNemar 6 / 5, p = 1,000), dan interval selisihnya [−0,036; +0,048] berpusat di sekitar nol.
- Keunggulan FINAL atas F0 (+3,6 poin) dan atas naive (+5,4 poin) juga **tidak signifikan** setelah koreksi Holm (p = 0,633 untuk keduanya). Kesimpulan Keputusan 44 tidak berubah: berita menambah akurasi secara nominal di atas histori harga, tetapi tidak signifikan pada test 167 hari.
- Histori harga tetap merupakan sumber informasi utama, dan teks tetap tidak mampu memprediksi tanpa harga.
- Menambah fitur teks tidak menaikkan akurasi: F1 (33 fitur) 0,760, FINAL (89 fitur) 0,766, kitchen sink (189 fitur) 0,737.
- **Akurasi valid bukan penunjuk yang andal untuk urutan test pada skala selisih ini.** Run dengan valid tertinggi dari seluruh eksperimen (F1 + RoBERTa fulltext + NER, valid 0,830) hanya mencapai test 0,760. Karena itu urutan antar run di test tidak ditafsirkan, dan aturan pemilihan yang ditetapkan sebelum eksperimen (Keputusan 55) dipegang apa adanya.
- Karena FINAL tidak terbukti lebih baik daripada F1, sedangkan F1 jauh lebih sederhana (33 vs 89 fitur, tanpa inferensi model bahasa), **F1 tetap layak dipertahankan sebagai model utama**. Early fusion versi penuh dilaporkan sebagai hasil negatif yang jujur.

### Keterbatasan
- Test kecil (167 hari): satu hari setara 0,6 poin akurasi. Lebar interval bootstrap akurasi sekitar ±6 poin, jauh lebih besar daripada selisih antar model. Uji ini hanya mampu mendeteksi selisih yang besar; "tidak signifikan" berarti tidak terbukti, bukan terbukti tidak ada.
- Sebanyak 41 run dievaluasi pada test yang sama. Hanya FINAL yang diklaim, tetapi angka run lain tetap terlihat, sehingga angka tertinggi di antara run analisis (mis. 0,784) hampir pasti dibesar-besarkan oleh kebetulan.
- Valid juga kecil (165 hari) dan dipakai dua kali: memilih hyperparameter dari 27 kombinasi dan memilih model final dari enam run FULL.
- Evaluasi hanya pada satu periode test (Desember 2025 – September 2026).
- Jumlah komponen PCA (20) ditetapkan tanpa validasi (Keputusan 48); hasil dengan jumlah komponen lain tidak diketahui.
- LDA train-only di-fit satu kali dengan satu seed; variasi antar seed LDA tidak diukur.
- Multi-seed tidak memberi informasi ketidakpastian (std = 0), sehingga variasi akibat pelatihan tidak terukur.
- Feature importance bawaan XGBoost dipengaruhi jumlah kolom per blok dan korelasi antar fitur, sehingga hanya dipakai sebagai gambaran, bukan ukuran kontribusi.
- Keterbatasan Keputusan 44 tetap berlaku (rancangan fitur harga, kenaikan jumlah artikel per hari pada 2025–2026).
