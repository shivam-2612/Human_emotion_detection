# Face Emotion Detection Project

This project detects human facial emotions in real-time using two approaches: **DeepFace with Haar Cascade** and a **Custom Trained CNN Model**.

---

## 📂 Project Structure

### 1. **Method 1: Using DeepFace**
- `emotion.py` →  
  - Uses **DeepFace** library and **Haar Cascade Classifier** to detect faces and predict emotions in real-time.  
  - Simple and fast, does not require custom model training.  
  - Detects emotions such as: *happy, sad, angry, surprised, neutral, disgust, fear*.  

### 2. **Method 2: Using Custom Trained Model**
- `trainmodel.ipynb` →  
  - Jupyter notebook where the model is trained on the **Kaggle Face Expression Recognition Dataset** ([link](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)).  
  - Preprocesses images, builds and trains a CNN model, and saves:
    - `emotiondetectmodel.h5` → Trained model weights.  
    - `emotiondetectmodel.json` → Model architecture.  

- `realtimedetection.py` →  
  - Loads the trained model (`.h5` + `.json`) and predicts emotions in real-time.  
  - Displays the emotion on detected faces in live webcam feed.

**Trained Model Download**  
  - If you do not want to train the model yourself, you can download the pre-trained model files from this link:  
    [Download Trained Model](https://drive.google.com/drive/folders/17gYQ7j7hjJ4A09jGw_L3BD7lquKhWir0?usp=sharing)

---

## ⚙️ Setup Instructions

1. **Install Python**  
   - Tested with **Python 3.9+**.  

2. **Clone the Repository**
   ```bash
   git clone <https://github.com/shivam-2612/Human_emotion_detection.git>
   cd face-emotion-detection
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Scripts**

* **DeepFace + Haar Cascade**

  ```bash
  python emotion.py
  ```
* **Custom Trained Model**

  ```bash
  python realtimedetection.py
  ```

---

## 📚 Libraries Used

* OpenCV (`cv2`)
* DeepFace
* TensorFlow / Keras
* Numpy
* Matplotlib (for training visualization)
* JSON (for saving/loading model architecture)

---

## 📝 Notes

* Ensure you have a working webcam for real-time detection.
* DeepFace method is faster and easier to implement but relies on pre-trained models.
* Custom trained CNN gives flexibility to retrain on your own datasets and can achieve higher accuracy for your dataset.
* Training may require GPU for faster results, especially with large datasets.

---

## 🔗 Dataset

* Kaggle: [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

---

## ⚡ Recommendations

* Run `emotion.py` for quick testing of emotion detection.
* Train your own model in `trainmodel.ipynb` to improve detection accuracy for your specific use case.
* Or use the pre-trained model from the Google Drive link to skip training.
