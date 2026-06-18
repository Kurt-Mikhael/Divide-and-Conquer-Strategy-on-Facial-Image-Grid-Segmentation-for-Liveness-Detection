# Liveness Detection: Divide & Conquer vs Lightweight CNN

**Tujuan:** Membandingkan performa algoritma Divide & Conquer (heuristic pre-filter) dengan model Deep Learning ringan (Lightweight CNN) untuk liveness detection pada CPU. Fokus utama adalah optimasi waktu komputasi dan akurasi deteksi.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Struktur Folder](#struktur-folder)
3. [Step-by-Step: Cara Menjalankan](#step-by-step-cara-menjalankan)
   - [Step 0: Persiapan Dataset](#step-0-persiapan-dataset)
   - [Step 1: Training Model CNN](#step-1-training-model-cnn)
   - [Step 2: Benchmark Divide &amp; Conquer](#step-2-benchmark-divide--conquer)
   - [Step 3: Perbandingan D&amp;C vs CNN](#step-3-perbandingan-dc-vs-cnn)
   - [Step 4: Generate Visualisasi Grid](#step-4-generate-visualisasi-grid)
4. [Penjelasan File &amp; Script](#penjelasan-file--script)
5. [Hasil yang Dihasilkan](#hasil-yang-dihasilkan)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Dependensi yang Dibutuhkan

```bash
# Python 3.8+ (disarankan Python 3.13)
# Install dependensi dengan:
pip install opencv-python numpy matplotlib torch Pillow
```

**Library yang digunakan:**

- `opencv-python` (cv2) - Image processing dan variance calculation
- `numpy` - Array operations dan statistik
- `matplotlib` - Visualisasi grafik dan segmentasi
- `torch` - Deep Learning framework (PyTorch)
- `Pillow` - Image loading untuk training

**Catatan:**

- Project ini **TIDAK membutuhkan GPU** (CUDA). Semua training dan inference berjalan di CPU.
- `torchvision` **tidak dibutuhkan**. Implementasi dataset custom sudah dibuat dengan OpenCV dan Pillow.

### Periksa Dependensi

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import PIL; print('Pillow:', PIL.__version__)"
```

---

## Struktur Folder

```
C:\Users\Asus\Image Detection\divide-and-conquer\
│
├── liveness_detection/              # Main project folder
│   ├── datasets/                    # Dataset gambar
│   │   ├── real/                    # 587 gambar wajah asli
│   │   └── spoof/                   # 609 gambar wajah palsu
│   │
│   ├── models/                      # Model definitions
│   │   ├── __init__.py              # Data classes (GridResult, ProcessingResult)
│   │   ├── dl_cnn.py                # Lightweight CNN model (PyTorch)
│   │   └── saved/                   # Hasil training model (best_model.pth)
│   │
│   ├── segmentation/                # Algoritma segmentasi
│   │   └── segmenters.py            # DivideAndConquerSegmenter, NaiveFullProcessor
│   │
│   ├── detection/                   # Algoritma deteksi
│   │   └── detectors.py             # VarianceBasedDetector, ThresholdLivenessDetector
│   │
│   ├── strategies/                  # Variance calculation strategies
│   │   └── variance_calculators.py  # Laplacian, Sobel, LBP
│   │
│   ├── processing/                  # Pipeline processing
│   │   └── pipeline.py              # DatasetProcessor, BenchmarkRunner
│   │
│   ├── utils/                       # Utility functions
│   │   └── visualizer.py            # Visualizer, Timer, ConsoleLogger
│   │
│   ├── training/                    # Training scripts
│   │   └── train_dl.py              # Script training CNN
│   │
│   ├── main.py                      # Entry point utama D&C
│   ├── comparison_dl.py             # Script perbandingan D&C vs CNN
│   ├── generate_segmentation_grid.py # Script visualisasi grid
│   └── interfaces/                  # Abstract interfaces (OOP)
│
└── results/                         # Folder hasil output
    ├── comparison/                  # Hasil perbandingan (20 gambar + report)
    └── segmentation_grid/           # Visualisasi grid (4 file)
```

---

## Step-by-Step: Cara Menjalankan

### Step 0: Persiapan Dataset

**Struktur dataset yang diharapkan:**
```
liveness_detection/datasets/
├── real/
│   ├── real_001.jpg
│   ├── real_002.jpg
│   └── ... (587 gambar)
└── spoof/
    ├── fake_001.jpg
    ├── fake_002.jpg
    └── ... (609 gambar)
```

**Jika Anda ingin menggunakan dataset sendiri:**
1. Siapkan folder `liveness_detection/datasets/`
2. Buat 2 subfolder: `real/` dan `spoof/`
3. Masukkan gambar wajah asli ke `real/`
4. Masukkan gambar wajah palsu (foto, screen) ke `spoof/`

**Format yang didukung:** `.jpg`, `.jpeg`, `.png`, `.bmp`

> **Catatan:** Dataset sudah tersedia di project ini (1196 gambar total). Jika sudah ada, langsung lanjut ke Step 1.

---

### Jalankan Semua Proses dari main.py

**Semua proses sekarang dijalankan dari satu file utama:**

```bash
cd "C:\Users\Asus\Image Detection\divide-and-conquer"

# Jalankan pipeline lengkap
python liveness_detection/main.py
```

**Script ini akan menjalankan 4 step secara otomatis:**

#### Step 1: Training CNN (otomatis jika model belum ada)
- Melatih model Lightweight CNN dengan 2.1 juta parameter
- Training di CPU, 5 epochs, ~25 detik
- Akurasi validasi: ~90-94%
- Output: `liveness_detection/models/saved/best_model.pth`

**Catatan:** Jika model sudah ada, training akan di-skip. Untuk retraining, hapus file `best_model.pth`.

#### Step 2: Benchmark Divide & Conquer
- Memproses SEMUA gambar di dataset untuk benchmark
- Menjalankan perbandingan metode (D&C vs Naive Full Process)
- Menganalisis kompleksitas waktu (O(N log N))
- Output: `results/benchmark_report.txt`, `results/complexity_analysis.png`

#### Step 3: Perbandingan D&C vs CNN (hanya 10 sampel)
- Memilih 10 gambar random dari real dan 10 dari spoof
- Menjalankan D&C dan CNN pada setiap sampel
- Mengukur waktu eksekusi dan akurasi
- Membuat visualisasi perbandingan side-by-side
- Output: `results/comparison/` (20 gambar + report + charts)

#### Step 4: Generate Visualisasi Grid
- Mengambil hasil perbandingan dari Step 3
- Membuat grid ringkas 2x5 untuk setiap label
- Output: `results/segmentation_grid/` (4 file)

**Contoh output:**
```
================================================================================
LIVENESS DETECTION SYSTEM
Divide & Conquer vs Lightweight CNN
================================================================================

Dataset: liveness_detection\datasets
Model: liveness_detection\models\saved\best_model.pth

Model found: liveness_detection\models\saved\best_model.pth
Skipping training. To retrain, delete the model file.

============================================================
STEP 2: BENCHMARK DIVIDE & CONQUER
============================================================
...

============================================================
STEP 3: DIVIDE & CONQUER vs LIGHTWEIGHT CNN COMPARISON
============================================================

Selected 10 real and 10 spoof images

[1/10] real_5502.jpg
  D&C: 2.0ms, grids=13, live=True
  CNN: 2.5ms, live=True

...

Method               Accuracy     Avg Time (ms)   Std Time (ms)
-----------------------------------------------------------------
Divide & Conquer     50.0%       1.74          0.09
Lightweight CNN      90.0%       2.61          1.66

============================================================
STEP 4: GENERATE SEGMENTATION GRID
============================================================
Found 10 real and 10 spoof comparison images
Compact grid saved: results\segmentation_grid\real_10_samples_grid.png
Compact grid saved: results\segmentation_grid\spoof_10_samples_grid.png
Full panel saved: results\segmentation_grid\real_full_panel.png
Full panel saved: results\segmentation_grid\spoof_full_panel.png

================================================================================
ALL PROCESSES COMPLETE
================================================================================

Results saved to:
  results/benchmark_report.txt
  results/benchmark_comparison.png
  results/complexity_analysis.png
  results/comparison/ (20 comparison images + charts)
  results/segmentation_grid/ (4 grid visualizations)
================================================================================
```

---

### Output yang Dihasilkan

```
results/
├── benchmark_report.txt          # Report benchmark
├── benchmark_comparison.png      # Grafik perbandingan metode
├── complexity_analysis.png       # Analisis Big-O O(N log N)
├── comparison/
│   ├── real_01_comparison.png    # 10 gambar perbandingan real
│   ├── ...
│   ├── real_10_comparison.png
│   ├── spoof_01_comparison.png   # 10 gambar perbandingan spoof
│   ├── ...
│   ├── spoof_10_comparison.png
│   ├── comparison_charts.png     # Grafik perbandingan (4 panel)
│   ├── comparison_report.txt     # Report lengkap
│   └── metrics.json              # Data metrics
└── segmentation_grid/
    ├── real_10_samples_grid.png   # Grid ringkas 10 real
    ├── spoof_10_samples_grid.png  # Grid ringkas 10 spoof
    ├── real_full_panel.png        # Panel lengkap real
    └── spoof_full_panel.png       # Panel lengkap spoof
```

---

## Ringkasan Alur Kerja

```
┌─────────────────────────────────────────────────────┐
│  Step 0: Siapkan dataset (real/ & spoof/)            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Jalankan main.py (semua proses otomatis)           │
│  python liveness_detection/main.py                 │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Step 1: Training CNN (jika belum ada)      │   │
│  │  Output: models/saved/best_model.pth          │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Step 2: Benchmark D&C (semua image)        │   │
│  │  Output: results/benchmark_*.png             │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Step 3: Perbandingan D&C vs CNN (10+10)     │   │
│  │  Output: results/comparison/ (20 gambar)     │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Step 4: Generate Grid Visualisasi          │   │
│  │  Output: results/segmentation_grid/ (4 file) │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```
liveness_detection/datasets/
├── real/
│   ├── real_001.jpg
│   ├── real_002.jpg
│   └── ... (587 gambar)
└── spoof/
    ├── fake_001.jpg
    ├── fake_002.jpg
    └── ... (609 gambar)
```

**Jika asisten ingin menggunakan dataset sendiri:**

1. Siapkan folder `liveness_detection/datasets/`
2. Buat 2 subfolder: `real/` dan `spoof/`
3. Masukkan gambar wajah asli ke `real/`
4. Masukkan gambar wajah palsu (foto, screen) ke `spoof/`

**Format yang didukung:** `.jpg`, `.jpeg`, `.png`, `.bmp`

> **Catatan:** Dataset sudah tersedia di project ini (1196 gambar total). Jika sudah ada, langsung lanjut ke Step 1.

---

### 1. Training Model CNN

**Script:** `liveness_detection/training/train_dl.py`

**Apa yang dilakukan:**

- Melatih model Lightweight CNN dengan 2.1 juta parameter
- Menggunakan dataset real vs spoof
- Training di CPU, 5 epochs, ~25 detik
- Akurasi validasi: ~90-94%

**Cara menjalankan:**

```bash
# Navigate ke project root
cd "C:\Users\Asus\Image Detection\divide-and-conquer"

# Jalankan training
python liveness_detection/training/train_dl.py
```

**Output yang dihasilkan:**

```
liveness_detection/models/saved/
├── best_model.pth          # Model terbaik (berdasarkan val accuracy)
├── final_model.pth         # Model terakhir (epoch 5)
├── training_history.png    # Grafik loss dan accuracy
└── training_history.json   # Data history training
```

**Parameter yang bisa diubah:**

```python
# Di file train_dl.py, di bagian main():
model_path = train_model(
    dataset_dir=str(dataset_dir),
    output_dir=str(output_dir),
    epochs=5,        # Bisa naikkan ke 10-20 untuk akurasi lebih tinggi
    batch_size=32,
    lr=0.001,
    input_size=128,   # Ukuran input gambar (128x128)
    grayscale=False,  # True untuk grayscale (lebih cepat)
    val_split=0.2     # 20% data untuk validasi
)
```

**Contoh output training:**

```
============================================================
LIGHTWEIGHT CNN TRAINING
============================================================
Found classes: ['real', 'spoof']
Using device: cpu

Model Summary:
  Total parameters: 2,121,346
  Model size: 8.09 MB (float32)

Dataset: 1196 total
  Train: 957, Val: 239

Epoch 1/5 | Train Loss: 1.1975, Train Acc: 0.7085 | Val Loss: 0.4749, Val Acc: 0.7657
Epoch 2/5 | Train Loss: 0.3062, Train Acc: 0.8798 | Val Loss: 0.2667, Val Acc: 0.8954
Epoch 3/5 | Train Loss: 0.2092, Train Acc: 0.9154 | Val Loss: 0.2421, Val Acc: 0.8996
Epoch 4/5 | Train Loss: 0.1903, Train Acc: 0.9216 | Val Loss: 0.1841, Val Acc: 0.9372
Epoch 5/5 | Train Loss: 0.1532, Train Acc: 0.9310 | Val Loss: 0.2145, Val Acc: 0.9079

Training completed in 25.43s
Best validation accuracy: 0.9372
```

---

### 2. Benchmark Divide & Conquer

**Script:** `liveness_detection/main.py`

**Apa yang dilakukan:**

- Memproses semua gambar di dataset menggunakan D&C
- Membuat visualisasi segmentasi grid untuk setiap gambar
- Menjalankan benchmark perbandingan metode (D&C vs Naive Full Process)
- Menganalisis kompleksitas waktu (O(N log N))

**Cara menjalankan:**

```bash
cd "C:\Users\Asus\Image Detection\divide-and-conquer"

# Jalankan benchmark D&C
python liveness_detection/main.py
```

**Output yang dihasilkan:**

```
results/
├── benchmark_report.txt       # Report benchmark
├── benchmark_comparison.png   # Grafik perbandingan metode
├── complexity_analysis.png    # Analisis Big-O time complexity
├── grid_segmentation.png      # Contoh visualisasi grid
└── variance_distribution.png  # Distribusi variance
```

**Catatan:**

- Script ini akan membuat folder `results/` jika belum ada.
- Hasil benchmark berguna untuk makalah (membuktikan O(N log N)).

---

### 3.Perbandingan D&C vs CNN

**Script:** `liveness_detection/comparison_dl.py`

**Apa yang dilakukan:**

- Memilih 10 gambar random dari setiap label (real & spoof)
- Menjalankan D&C dan CNN pada setiap gambar
- Mengukur waktu eksekusi dan akurasi
- Membuat visualisasi perbandingan side-by-side

**Cara menjalankan:**

```bash
cd "C:\Users\Asus\Image Detection\divide-and-conquer"

# Jalankan perbandingan (WAJIB sudah training CNN di Step 1)
python liveness_detection/comparison_dl.py
```

**Output yang dihasilkan:**

```
results/comparison/
├── real_01_comparison.png    # Perbandingan sampel real #1
├── real_02_comparison.png    # ... sampai real_10
├── ...
├── real_10_comparison.png
├── spoof_01_comparison.png   # Perbandingan sampel spoof #1
├── spoof_02_comparison.png   # ... sampai spoof_10
├── ...
├── spoof_10_comparison.png
├── comparison_charts.png     # Grafik perbandingan (4 panel)
├── comparison_report.txt     # Report lengkap
└── metrics.json              # Data metrics (untuk analisis)
```

**Contoh output:**

```
================================================================================
DIVIDE & CONQUER vs LIGHTWEIGHT CNN COMPARISON
================================================================================

Selected 10 real and 10 spoof images

[1/10] real_5502.jpg
  D&C: 2.0ms, grids=13, live=True
  CNN: 9.8ms, live=True

...

Method               Accuracy     Avg Time (ms)   Std Time (ms)
-----------------------------------------------------------------
Divide & Conquer     50.0%       1.74          0.09
Lightweight CNN      90.0%       2.61          1.66

Divide & Conquer:
  Real Accuracy:  9/10 (90.0%)
  Spoof Accuracy: 1/10 (10.0%)

Lightweight CNN:
  Real Accuracy:  9/10 (90.0%)
  Spoof Accuracy: 9/10 (90.0%)
```

---

### 4. Generate Visualisasi Grid

**Script:** `liveness_detection/generate_segmentation_grid.py`

**Apa yang dilakukan:**

- Mengambil hasil perbandingan dari Step 3
- Membuat grid 2x5 yang ringkas untuk setiap label
- Membuat panel lengkap yang menampilkan semua perbandingan

**Cara menjalankan:**

```bash
cd "C:\Users\Asus\Image Detection\divide-and-conquer"

# Jalankan visualisasi grid (WAJIB sudah jalankan Step 3)
python liveness_detection/generate_segmentation_grid.py
```

**Output yang dihasilkan:**

```
results/segmentation_grid/
├── real_10_samples_grid.png     # Grid 10 sampel real (D&C vs CNN)
├── spoof_10_samples_grid.png    # Grid 10 sampel spoof (D&C vs CNN)
├── real_full_panel.png          # Panel lengkap real
└── spoof_full_panel.png         # Panel lengkap spoof
```

---

## Ringkasan Alur Kerja

```
┌─────────────────────────────────────────────────────┐
│  Step 0: Siapkan dataset (real/ & spoof/)            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: Training CNN                                 │
│  python liveness_detection/training/train_dl.py       │
│  Output: models/saved/best_model.pth                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Benchmark D&C (opsional)                    │
│  python liveness_detection/main.py                   │
│  Output: results/benchmark_*.png                      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Perbandingan D&C vs CNN                     │
│  python liveness_detection/comparison_dl.py           │
│  Output: results/comparison/ (20 gambar + report)    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: Generate Grid Visualisasi                   │
│  python liveness_detection/generate_segmentation_grid.py
│  Output: results/segmentation_grid/ (4 file)          │
└─────────────────────────────────────────────────────┘
```

---

## Penjelasan File & Script

### 1. `liveness_detection/main.py`
**Entry point utama untuk menjalankan SELURUH pipeline.**

Integrasi 4 step dalam satu script:
- **Step 1:** Training CNN (otomatis jika model belum ada)
- **Step 2:** Benchmark D&C (memproses semua gambar untuk statistik)
- **Step 3:** Perbandingan D&C vs CNN (hanya 10 sampel real + 10 spoof)
- **Step 4:** Generate grid visualisasi segmentasi

**Cara jalankan:**
```bash
python liveness_detection/main.py
```

**Catatan:** Visualisasi segmentasi grid hanya dihasilkan untuk 10 sampel real dan 10 sampel spoof (bukan semua gambar di dataset).

### 2. `liveness_detection/training/train_dl.py`
Script training model Lightweight CNN (standalone).
- Dapat dijalankan sendiri untuk training
- Juga diintegrasikan ke main.py
- Menggunakan PyTorch pure (tanpa torchvision)
- Custom dataset dengan OpenCV/Pillow

### 3. `liveness_detection/comparison_dl.py`
Script perbandingan komprehensif (standalone).
- Dapat dijalankan sendiri untuk perbandingan
- Juga diintegrasikan ke main.py
- Select 10 random samples dari masing-masing label
- Generate visualisasi perbandingan

### 4. `liveness_detection/generate_segmentation_grid.py`
Script visualisasi konsolidasi (standalone).
- Dapat dijalankan sendiri untuk grid
- Juga diintegrasikan ke main.py
- Buat grid ringkas 2x5 untuk makalah

### 5. `liveness_detection/models/dl_cnn.py`

Definisi model Lightweight CNN.

- `LightweightCNN`: Arsitektur CNN (Conv2d 16→32→64)
- `SimpleCNNDetector`: Wrapper untuk preprocessing
- `CNNFullImageDetector`: Interface untuk pipeline
- `CNNInferenceSegmenter`: Adapter untuk segmentasi
- `LightweightCNNTrainer`: Utility training

### 6. `liveness_detection/segmentation/segmenters.py`

Implementasi segmentasi.

- `DivideAndConquerSegmenter`: Recursive quad-tree segmentation
- `NaiveFullProcessor`: Proses gambar utuh (baseline)
- `SlidingWindowSegmenter`: Sliding window (baseline)

### 7. `liveness_detection/detection/detectors.py`

Implementasi deteksi.

- `VarianceBasedDetector`: Deteksi berbasis rata-rata variance
- `ThresholdLivenessDetector`: Threshold sederhana
- `AnomalyLivenessDetector`: Deteksi anomali
- `EnsembleLivenessDetector`: Ensemble voting

### 8. `liveness_detection/strategies/variance_calculators.py`

Strategi perhitungan variance.

- `LaplacianVarianceCalculator`: Laplacian variance (default)
- `SobelVarianceCalculator`: Gradient magnitude
- `CombinedVarianceCalculator`: Kombinasi weighted
- `LocalBinaryPatternVariance`: LBP histogram

---

## Hasil yang Dihasilkan

### Metrik Perbandingan (D&C vs CNN)

| Metode           | Akurasi | Waktu Rata-rata | Real Acc | Spoof Acc |
| ---------------- | ------- | --------------- | -------- | --------- |
| Divide & Conquer | 50%     | 1.74 ms         | 90%      | 10%       |
| Lightweight CNN  | 90%     | 2.61 ms         | 90%      | 90%       |

### Interpretasi

- **D&C lebih cepat** (~1.5x) karena memangkas area background
- **CNN lebih akurat** (90% vs 50%) karena belajar fitur kompleks
- **D&C cocok sebagai pre-filter** untuk memangkas data tidak relevan
- **CNN cocok untuk klasifikasi akhir** dengan akurasi tinggi

### File Visualisasi untuk Makalah

**Untuk Bab Eksperimen:**

1. `results/comparison/comparison_charts.png` - Grafik perbandingan 4 panel
2. `results/comparison/comparison_report.txt` - Report numerik
3. `results/segmentation_grid/real_10_samples_grid.png` - Grid 10 real
4. `results/segmentation_grid/spoof_10_samples_grid.png` - Grid 10 spoof

**Untuk Bab Analisis Kompleksitas:**

1. `results/complexity_analysis.png` - Grafik Big-O O(N log N)
2. `results/benchmark_comparison.png` - Perbandingan 3 metode

**Untuk Bab Metodologi:**

1. `results/comparison/real_01_comparison.png` - Contoh visualisasi D&C
2. `results/comparison/spoof_01_comparison.png` - Contoh hasil CNN

---

## Troubleshooting

### 1. ModuleNotFoundError: No module named 'liveness_detection'

**Penyebab:** Python tidak menemukan module liveness_detection.

**Solusi:**

```bash
# Pastikan Anda berada di root project
cd "C:\Users\Asus\Image Detection\divide-and-conquer"

# Pastikan Python path benar
python -c "import sys; print(sys.path)"

# Jika masih error, tambahkan path manual:
set PYTHONPATH=%PYTHONPATH%;C:\Users\Asus\Image Detection\divide-and-conquer
```

### 2. RuntimeError: Input type (double) and bias type (float) should be the same

**Penyebab:** Tipe data tensor tidak sesuai (float64 vs float32).

**Solusi:** Sudah diperbaiki di `dl_cnn.py`. Jika masih terjadi, pastikan model menggunakan `.float()`:

```python
tensor = torch.from_numpy(image).unsqueeze(0).float()
```

### 3. Dataset tidak ditemukan

**Penyebab:** Folder `datasets/real/` atau `datasets/spoof/` kosong/tidak ada.

**Solusi:**

```bash
# Periksa struktur folder
dir liveness_detection\datasets\real /s
dir liveness_detection\datasets\spoof /s

# Jika kosong, tambahkan gambar:
# - real/ : gambar wajah asli
# - spoof/ : gambar wajah palsu (foto, screen, mask)
```

### 4. Model tidak ditemukan (saat menjalankan comparison)

**Penyebab:** Belum training CNN atau path model salah.

**Solusi:**
```bash
# Periksa apakah model sudah di-training
ls liveness_detection\models\saved\best_model.pth

# Jika tidak ada, jalankan main.py (akan training otomatis):
python liveness_detection\main.py

# Atau jalankan training standalone:
python liveness_detection\training\train_dl.py
```

### 5. Memory Error / Out of Memory

**Penyebab:** Dataset terlalu besar untuk memori.

**Solusi:**

```python
# Di train_dl.py, kecilkan batch_size:
batch_size=16  # default: 32

# Atau kecilkan input_size:
input_size=64  # default: 128
```

### 6. Matplotlib tidak bisa save figure

**Penyebab:** Backend matplotlib tidak sesuai.

**Solusi:** Sudah diatur di code:

```python
matplotlib.use('Agg')  # Non-interactive backend
```

---

## Tips untuk Makalah IEEE

1. **Gunakan grafik dari `complexity_analysis.png`** untuk membuktikan O(N log N)
2. **Gunakan `comparison_charts.png`** untuk menunjukkan perbandingan metode
3. **Gunakan grid visualisasi** untuk menunjukkan hasil segmentasi
4. **Fokus pada kontribusi:** D&C mengoptimasi waktu komputasi, bukan akurasi
5. **Panjang makalah:** 6-10 halaman sesuai template IEEE
6. **Deadline:** 19 Juni 2026, 23:59 WIB

---

## Kontak & Referensi

- **Course:** IF2211 - Strategi Algoritma
- **Topik:** Divide and Conquer pada Computer Vision
- **Framework:** PyTorch, OpenCV, NumPy, Matplotlib
- **Hardware:** CPU (Intel/AMD) - tidak membutuhkan GPU
