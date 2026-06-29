# 🖐️ Camera Blur App

Aplikasi kamera real-time yang mengaktifkan efek **blur** secara otomatis ketika mendeteksi **2 jari** yang terangkat — baik telapak menghadap kamera maupun dibalik (punggung tangan).

Dibangun menggunakan Python, OpenCV, dan MediaPipe.

---

## ✨ Fitur

- Deteksi tangan real-time menggunakan MediaPipe
- Blur otomatis aktif saat tepat **2 jari** terangkat
- Akurat untuk **dua orientasi tangan**: telapak ke depan maupun punggung tangan ke depan
- Ringan dan berjalan di CPU biasa

---

## 🖥️ Requirements

- Python **3.8 – 3.11**
- Webcam / kamera laptop

> ⚠️ Gunakan Python 3.11 ke bawah. MediaPipe 0.10.9 belum mendukung Python 3.12+.

---

## ⚙️ Instalasi

**1. Clone repository ini**

```bash
git clone https://github.com/Dikicandra98/Camera-Blur.git
cd Camera-Blur
```

**2. (Opsional tapi disarankan) Buat virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install opencv-python mediapipe==0.10.9 numpy
```

---

## ▶️ Cara Menjalankan

```bash
python camera_blur.py
```

---

## 🎮 Kontrol

| Gesture / Tombol | Aksi |
|---|---|
| ✌️ Angkat 2 jari | Kamera menjadi blur |
| Jari lainnya (1, 3, 4, 5) | Kamera normal |
| `Q` atau `ESC` | Keluar dari aplikasi |

---

## 📁 Struktur Project

```
Camera-Blur/
├── camera_blur.py   # File utama aplikasi
└── README.md
```

---


---

## 🛠️ Cara Kerja

1. Setiap frame dari kamera diproses oleh **MediaPipe Hands** untuk mendeteksi 21 landmark titik tangan.
2. Orientasi telapak dideteksi menggunakan **cross product 3D** dari vektor pergelangan → MCP telunjuk dan pergelangan → MCP kelingking.
3. Berdasarkan orientasi tersebut, setiap jari diperiksa apakah terangkat atau tidak menggunakan perbandingan posisi ujung jari (TIP) terhadap sendi tengah (PIP).
4. Jika total jari terangkat = 2, frame dikenai **Gaussian Blur**.

---

## 📦 Dependencies

| Package | Versi | Fungsi |
|---|---|---|
| `opencv-python` | latest | Akses kamera & tampilan frame |
| `mediapipe` | 0.10.9 | Deteksi & tracking tangan |
| `numpy` | latest | Kalkulasi vektor & cross product |

---

## 📄 Lisensi

MIT License — bebas digunakan dan dimodifikasi.
