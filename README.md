<h1>🌾 Monitoring Crop Health using Deep Learning and Grad-CAM</h1><br>
<b>📌 Project Objective</b>:<br>
<br>This project simulates aerial crop health monitoring using deep learning and computer vision. The aim is to classify each region of a crop image as either healthy or unhealthy and highlight the affected areas using Grad-CAM overlays.<br>
<br>
<b>🧠 Models Used</b><br><br>
<b>✅ ResNet50 </b>– Deep residual CNN for classification<br>
ResNet-50 consists of 50 layers that are divided into 5 blocks, each containing a set of residual blocks. The residual blocks allow for the preservation of information from earlier layers, which helps the network to learn better representations of the input data.
<br>
The following are the main components of ResNET.
<br><br>
1. Convolutional Layers
The first layer of the network is a convolutional layer that performs convolution on the input image. This is followed by a max-pooling layer that downsamples the output of the convolutional layer. The output of the max-pooling layer is then passed through a series of residual blocks.
<br><br>
2. Residual Blocks
Each residual block consists of two convolutional layers, each followed by a batch normalization layer and a rectified linear unit (ReLU) activation function. The output of the second convolutional layer is then added to the input of the residual block, which is then passed through another ReLU activation function. The output of the residual block is then passed on to the next block.
<br><br>
3. Fully Connected Layer
The final layer of the network is a fully connected layer that takes the output of the last residual block and maps it to the output classes. The number of neurons in the fully connected layer is equal to the number of output classes.

<b>✅ EfficientNetB0</b> – Lightweight model optimized for performance<br>
EfficientNetB0 is the baseline model of the EfficientNet family, developed by Google. It balances model accuracy and efficiency using a novel technique called compound scaling, which scales depth, width, and resolution systematically.
<br>
🔧 Key Components:
MBConv Block (Mobile Inverted Bottleneck):
Combines several modern techniques for efficient convolution.
<br>
1. Expansion Layer:
Uses a 1x1 convolution to expand input channels, enabling richer feature learning.
<br>
2. Depthwise Convolution (3x3):
Applies one filter per input channel to extract spatial patterns efficiently.
<br>
3. Squeeze-and-Excitation:
Learns to recalibrate channel-wise responses:
<br>
Squeeze: Global average pooling.
<br>
Excitation: Learns weights to emphasize informative channels.
<br>
4. Projection Layer:
Uses a 1x1 convolution to reduce channel dimensions back to target size.<br><br>

<b>✅ Grad-CAM</b> – Used to visualize important regions contributing to the model's decision<br>
The gradient-weighted class activation map (Grad CAM) produces a heat map that highlights important regions of an image using the target gradients (dog, cat) of the final convolutional layer.
<br>
The Grad CAM method is a popular visualisation technique that is useful for understanding how a convolutional neural network has been driven to make a classification decision. It is class-specific, meaning that it can produce a separate visualisation for each class present in the image.
<br>
In the event of a classification error, this method can be very useful for understanding where the problem lies in the convolutional network. It also makes the algorithm more transparent.
<br><br>
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
```
pip install -r requirements.txt<br>
```
<b>📁 Dataset</b><br>
Used an augmented plant disease dataset from Kaggle. [Dataset link] (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)<br>.<br>
(Datasetis already splited into train,validation and test datasets)<br>

Images categorized as healthy or unhealthy(38 classes).<br>

Dataset was resized(224,224), normalized.<br>

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
|Model|	Accuracy|	Params|	Suitable For Grad-CAM<br>|<br>
|ResNet50|	1.0 |	🔺 ~24M|	✅ Yes|<br>
|EfficientNetB0|	99.74 |	🔻 ~7M|	✅ Yes (lightweight)|<br>

<b>🤝 Acknowledgements</b><br>
Dataset: Kaggle – New Plant Diseases Dataset (Augmented)<br>

TensorFlow, OpenCV, and Keras Teams<br>

<b>📬 Contact</b><br>
For questions or collaborations, contact [your email or GitHub profile].
