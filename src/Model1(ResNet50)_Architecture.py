import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def Conv_Block(x,output_channels):
    res=x
    
    x=Conv2D(output_channels//4,1,strides=(1,1),padding='valid')(x)
    x=BatchNormalization()(x)
    x=ReLU()(x)
    
    x=Conv2D(output_channels//4,3,strides=(1,1),padding='same')(x)
    x=BatchNormalization()(x)
    x=ReLU()(x)
    
    x=Conv2D(output_channels,1,strides=(1,1),padding='valid')(x)
    x=BatchNormalization()(x)

    #Shortcut projection
    res=Conv2D(output_channels,1,strides=(1,1),padding='valid')(res)
    res=BatchNormalization()(res)
    
    x=Add()([x,res])
    x=ReLU()(x)
    return x

def Identity_Block(x,output_channels):
    res=x
    
    x=Conv2D(output_channels//4,1,strides=(1,1),padding='valid')(x)
    x=BatchNormalization()(x)
    x=ReLU()(x)
    
    x=Conv2D(output_channels//4,3,strides=(1,1),padding='same')(x)
    x=BatchNormalization()(x)
    x=ReLU()(x)
    
    x=Conv2D(output_channels,1,strides=(1,1),padding='valid')(x)
    x=BatchNormalization()(x)

    x=Add()([x,res])  # no need os shortcut projection since identity block we used conv_block with same no. of output channels
    x=ReLU()(x)
    return x

def Build_ResNet50(input_shape=(224,224,3), num_classes=38):
    input_=Input(shape=input_shape)
    x=Conv2D(64, kernel_size=7, strides=2, padding='same')(input_)
    x=BatchNormalization()(x)
    x=ReLU()(x)
    x=MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding="same")(x)
    
    x=Conv_Block(x,256)
    x=Identity_Block(x,256)
    x=Identity_Block(x,256)
    
    x=Conv_Block(x,512)
    x=Identity_Block(x,512)
    x=Identity_Block(x,512)
    x=Identity_Block(x,512)
    
    x=Conv_Block(x,1024)
    x=Identity_Block(x,1024)
    x=Identity_Block(x,1024)
    x=Identity_Block(x,1024)
    x=Identity_Block(x,1024)
    x=Identity_Block(x,1024)
    
    x=Conv_Block(x,2048)
    x=Identity_Block(x,2048)
    x=Identity_Block(x,2048)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_,outputs=x)

resnet50=Build_ResNet50()
resnet50.summary()
