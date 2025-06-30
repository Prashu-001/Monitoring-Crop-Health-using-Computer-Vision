🌾 Monitoring Crop Health using Deep Learning and Grad-CAM
📌 Project Objective
This project simulates aerial crop health monitoring using deep learning and computer vision. The aim is to classify each region of a crop image as either healthy or unhealthy and highlight the affected areas using Grad-CAM overlays.

🧠 Models Used
✅ ResNet50 – Deep residual CNN for classification

✅ EfficientNetB0 – Lightweight model optimized for performance

✅ Grad-CAM – Used to visualize important regions contributing to the model's decision

🗃️ Project Structure
bash
Copy
Edit
Monitoring-Crop-Health-using-Computer-Vision/
├── data/               # Sample input images
├── models/             # Saved model files (.h5 or .keras)
├── notebooks/          # Training and inference notebooks
│   ├── training.ipynb
│   └── inference_gradcam.ipynb
├── src/                # Source code scripts
│   ├── model_builder.py
│   ├── train.py
│   ├── gradcam_utils.py
│   └── inference.py
├── outputs/            # Grad-CAM outputs and overlayed images
├── requirements.txt    # All dependencies
└── README.md
📦 Requirements
Python ≥ 3.8

TensorFlow ≥ 2.8

OpenCV

NumPy

Matplotlib

scikit-learn

Install with:

bash
Copy
Edit
pip install -r requirements.txt
📁 Dataset
Used an augmented plant disease dataset from Kaggle.

Images categorized as healthy or unhealthy.

Dataset was resized, normalized, and split into train/test sets.

🛠️ Model Training
Models were built using the Keras API in TensorFlow.

Both ResNet50 and EfficientNetB0 were trained on the dataset.

Training metrics (accuracy & loss) were tracked using matplotlib.

📄 Output:

Trained model saved as .h5 format under /models/.

🎯 Inference & Visualization
The trained model was used to predict crop health from test images.

Grad-CAM was used to highlight diseased regions:

Red overlays show unhealthy vegetation.

Output images show both original and heatmap overlays.

📂 Output:

Saved visualizations in /outputs/ as .jpg images.

Optionally, calculated the % area affected by disease per image.

📸 Sample Output
Original	Grad-CAM Heatmap	Overlay

📈 Results
Model	Accuracy	Params	Suitable For Grad-CAM
ResNet50	✅ High	🔺 ~25M	✅ Yes
EfficientNetB0	✅ Good	🔻 ~5M	✅ Yes (lightweight)

🚀 Run the Project
Train the model:
bash
Copy
Edit
python src/train.py --model resnet50 --epochs 20
Inference & Grad-CAM visualization:
bash
Copy
Edit
python src/inference.py --model models/resnet50.h5 --input data/test/
🤝 Acknowledgements
Dataset: Kaggle – New Plant Diseases Dataset (Augmented)

TensorFlow, OpenCV, and Keras Teams

📬 Contact
For questions or collaborations, contact [your email or GitHub profile].
