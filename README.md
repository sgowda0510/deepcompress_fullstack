# DeepCompress — Full Stack Model Compression Platform

> Final Year B.E. Project · IEEE Conference Ready · Full Stack Web App

---

## What is this?

A full-stack web platform for compressing deep learning models using Pruning and Quantization. Users can select a dataset, configure compression settings, run the pipeline, and see results on a live dashboard — all from the browser.

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start the web server
```bash
python app.py
```

### Step 3 — Open in browser
```
http://localhost:5000
```

---

## Supported Datasets

| Dataset    | Type    | Classes | Samples | Size   |
|------------|---------|---------|---------|--------|
| CIFAR-10   | Image   | 10      | 60,000  | 170 MB |
| MNIST      | Image   | 10      | 70,000  | 11 MB  |
| Diabetes   | Tabular | 2       | 768     | <1 MB  |
| CIFAR-100  | Image   | 100     | 60,000  | 170 MB |
| **Custom** | Tabular | Auto    | Auto    | Upload |

---

## Adding Custom Datasets (New Feature)

Use the **"➕ Insert Dataset"** tab in the web interface to upload any tabular `.csv` dataset:
1. Provide a dataset name and description.
2. Select your `.csv` file (target label in the final column).
3. Click **"Upload & Register Dataset"**.
4. The system automatically inspects rows, features, and classes, registers the dataset, and displays it in **Step 1** ready for training and compression!

---

## How to Run in VS Code

1. Open VS Code:
   ```powershell
   code "C:\Users\Shashank K C\Downloads\deepcompress_fullstack\deepcompress_fullstack"
   ```
2. Open the integrated terminal (`Ctrl + \`` or `Terminal -> New Terminal`).
3. Run the app:
   ```powershell
   & "C:\Users\Shashank K C\AppData\Local\Programs\Python\Python310\python.exe" app.py
   ```
4. Open your browser: `http://localhost:5000`

---

## CLI Usage (no web server)

```bash
# Run full pipeline
python main.py

# Skip training (use saved model)
python main.py --skip-train

# Use different dataset
python main.py --dataset mnist

# Use GPU
python main.py --device cuda
```

---

## Project Structure

```
deepcompress_fullstack/
├── app.py              ← Flask web server
├── main.py             ← CLI pipeline
├── train.py            ← Training
├── prune.py            ← L1 pruning
├── quantize.py         ← INT8 quantization
├── energy.py           ← CodeCarbon tracking
├── evaluate.py         ← Metrics evaluation
├── visualize.py        ← Matplotlib plots
├── config.py           ← All settings
├── models/             ← Saved .pt files
├── results/            ← experiment_results.csv
├── plots/              ← PNG charts
├── templates/
│   └── index.html      ← Frontend
├── static/
│   ├── css/style.css   ← Styles
│   └── js/app.js       ← JavaScript
└── requirements.txt
```

---

## Tech Stack

| Layer         | Technology                                         |
|---------------|----------------------------------------------------|
| 3D Twin & UI  | Three.js r128, WebGL, Chart.js 4.x, Glassmorphism  |
| Backend       | Python 3.10, Flask 3.x, Flask-CORS                 |
| Deep Learning | PyTorch 2.x, TorchVision, Dynamic Quantization     |
| Energy Engine | CodeCarbon (RAPL CPU registers & kWh tracking)    |
| Edge Targets  | Broadcom BCM2711 (ARM Cortex-A72 & NEON INT8 SIMD) |
| Data & Eval   | CIFAR-10, MNIST, Diabetes, CIFAR-100, Custom CSV   |

---

## Project Authors & Student Researchers (KNSIT)

* **Mr. Aditya Manjunath Pujar** — Final Year B.E. CSE (Data Science) · KNSIT · Batch 2027
* **Ms. Jhanavi R R** — Final Year B.E. CSE (Data Science) · KNSIT · Batch 2027
* **Ms. Nandini D R** — Final Year B.E. CSE (Data Science) · KNSIT · Batch 2027
* **Mr. Shashank K C** — Final Year B.E. CSE (Data Science) · KNSIT · Batch 2027

**Institution**: KNS Institute of Technology (KNSIT), Bengaluru
