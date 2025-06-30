import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
from ResNet50_Architecture import Build_resnet50
from EfficientNetB0_Architecture import Build_EfficientNetB0

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 38

def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label
  
def get_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
     ).map(preprocess).repeat().prefetch(tf.data.AUTOTUNE)
  
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )

    class_names = val_ds.class_names
    val_ds=val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def build_model(name):
    if name == 'resnet50':
        model= Build_ResNet5o()
    elif name == 'efficientnetb0':
        model=Build_EfficientNetB0()
    else:
        raise ValueError("Unsupported model. Use 'resnet50' or 'efficientnetb0'.")
      
    return model

def plot_training(history, output_dir):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_plot.png'))
    plt.close()

def main(args):
    train_ds, val_ds = get_data()
    model = build_model(args.model)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model.save(os.path.join(args.save_dir, f"{args.model}_crop_health_model.keras"))

    # Save training plot
    plot_training(history, args.save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train crop health classification model")
    parser.add_argument('--model', type=str, required=True, choices=['resnet50', 'efficientnetb0'],
                        help='Model architecture: resnet50 or efficientnetb0')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save model and plots')

    args = parser.parse_args()
    main(args)
