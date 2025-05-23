# 📄 Doc-Scanner

A Python-based command-line Document Scanner that uses traditional image processing techniques with OpenCV to detect, extract, and enhance documents from images.

---

## 🛠 Features

- Automatic document detection from an image.
- Perspective transformation to flatten the document.
- Adaptive thresholding to enhance readability.
- Simple CLI for image selection and output.
- Modular and easy to extend (e.g., U-Net integration in the future).

---

## 🖼 Example

Original Image → Perspective Transform → Thresholded Result  
<p align="center">
  <img src="upload/original.jpg" width="250">
  <img src="upload/scanned.jpg" width="250">
  <img src="upload/thresholded.jpg" width="250">
</p>

---

## 📁 Project Structure

```

Doc-Scanner/
├── app.py                  # Entry point CLI
├── scanner.py              # Core logic for scanning using OpenCV
├── unet\_model.py           # (Not used currently) U-Net model for future deep learning enhancement
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── samples/                # Example input/output images

````

---

## ⚙️ How It Works

1. **Image Preprocessing**: Resize and convert the image to grayscale.
2. **Edge Detection**: Use Canny edge detection to find document boundaries.
3. **Contour Detection**: Identify the largest 4-point contour (assumed to be the document).
4. **Perspective Transform**: Warp the image to get a top-down view.
5. **Thresholding**: Apply adaptive thresholding to enhance readability.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/Doc-Scanner.git
cd Doc-Scanner
````

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python app.py
```

* The script will prompt you to input the path to your image.
* It will process and display the scanned result.
* Press any key to save and close the window.

---

## 🧠 Future Work

* Integration of U-Net deep learning model (`unet_model.py`) for improved document detection in noisy or complex backgrounds.
* Add GUI support using Tkinter or PyQt.
* Support for batch processing and camera input.

---

## ✅ Dependencies

* OpenCV
* NumPy
* (Optional for future) TensorFlow / Keras

Install them using:

```bash
pip install opencv-python numpy
```

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 🙌 Acknowledgements

Inspired by various document scanner apps and tutorials. Future U-Net integration idea is based on semantic segmentation models used in computer vision.


