<h1>🌾 Monitoring Crop Health using Deep Learning and Grad-CAM</h1><br>
<b>📌 Project Objective</b>:<br>
<br>This project simulates aerial crop health monitoring using deep learning and computer vision. The aim is to classify each region of a crop image as either healthy or unhealthy and highlight the affected areas using Grad-CAM overlays.<br>
<br>
<b>🧠 Models Used</b><br><br>
<b>✅ ResNet50 </b>– Deep residual CNN for classification<br>

<b>✅ EfficientNetB0</b> – Lightweight model optimized for performance<br>

<b>✅ Grad-CAM</b> – Used to visualize important regions contributing to the model's decision<br>

<b>🗃️ Project Structure</b><br>
<br>
```
Monitoring-Crop-Health-using-Computer-Vision/
├── data/               # Sample input images
├── models/             # Saved model files (.h5)
├── notebooks/          # Training and inference notebooks
│   ├── EfficientNetB0-Training.ipynb
|   ├── ResNet50-Training.ipynb
│   └── inference_notebook.ipynb
├── src/                # Source code scripts
|   ├── Grad-CAM.py
│   ├── Model1(ResNet50_Architecture.py
|   ├── Model2(EfficientNetB0_Architecture.py
│   ├── train.py
│   └── inference.py
├── outputs/            # Grad-CAM outputs and overlayed images
├── requirements.txt    # All dependencies
└── README.md
```
<br>
<b>📦 Requirements</b><br>
Python <br>

TensorFlow <br>

OpenCV<br>

NumPy<br>

Matplotlib<br>

scikit-learn<br>

<b>Install with</b>:<br>
pip install -r requirements.txt<br>
<b>📁 Dataset</b><br>
Used an augmented plant disease dataset from Kaggle.<br>

Images categorized as healthy or unhealthy.<br>

Dataset was resized, normalized, and split into train/test sets.<br>

<b>🛠️ Model Training</b><br>
Models were built using the Keras API in TensorFlow.<br>

Both ResNet50 and EfficientNetB0 were trained on the dataset.<br>

Training metrics (accuracy & loss) were tracked using matplotlib.<br>

<b>📄 Output:</b><br>

Trained model saved as .h5 format under /models/.<br>

<b>🎯 Inference & Visualization</b><br>
The trained model was used to predict crop health from test images.<br>

Grad-CAM was used to highlight diseased regions:<br>

Red overlays show unhealthy vegetation.<br>

Output images show both original and heatmap overlays.<br>

##<b>📂 Output:</b><br>

Saved visualizations in /outputs/ as .jpg images.<br>
<br>
<b>📸 Sample Output</b><br><br>
![](outputs/gradcam_output_4.jpg) <br>
![](outputs/gradcam_output_19.jpg) <br>
![](outputs/gradcam_output_24.jpg) <br>

<b>📈 Results</b><br>
|Model|	Accuracy|	Params|	Suitable For Grad-CAM<br>|
|ResNet50|	1.0 |	🔺 ~24M|	✅ Yes|
|EfficientNetB0|	99.0 |	🔻 ~7M|	✅ Yes (lightweight)|

<b>🤝 Acknowledgements</b><br>
Dataset: Kaggle – New Plant Diseases Dataset (Augmented)<br>

TensorFlow, OpenCV, and Keras Teams<br>

<b>📬 Contact</b><br>
For questions or collaborations, contact [your email or GitHub profile].
