---
title: What is an AutoEncoder?
date: 2023-04-19 19:25:00 +02:00
categories: [AI]
tags: [deeplearning, machinelearning, autoencoder]
---

Autoencoders are neural networks that are widely used for unsupervised learning. They are used to learn a compressed representation of input data by encoding it into a latent space and then decoding it back into the original input space. Autoencoders are used in a variety of applications such as image and speech recognition, anomaly detection, and data compression.

In this blog post, we will explore the basics of autoencoders and implement them using TensorFlow.

## Introduction to Autoencoders

An autoencoder is made up of two parts: an encoder and a decoder. The encoder takes the input data and maps it to a lower-dimensional representation, while the decoder maps the lower-dimensional representation back to the original input space. The goal of an autoencoder is to learn a compressed representation of the input data that captures the most important features of the data.

<img src = "https://i.imgur.com/U9TeorA.png">

In order to train an autoencoder, we need to define a loss function that measures the difference between the input data and the reconstructed data. The loss function is typically a mean squared error function or a binary cross-entropy function.

Autoencoders can be used for various applications, such as:

- Data compression: Autoencoders can be used to compress data into a lower-dimensional representation, which can be stored more efficiently.
- Anomaly detection: Autoencoders can be used to detect anomalies in data by comparing the reconstruction error of the input data with a threshold.
- Image and speech recognition: Autoencoders can be used to learn a compressed representation of images and speech, which can be used for classification tasks.

## Implementing Autoencoders with TensorFlow

Now, let's implement autoencoders using TensorFlow. 

### Basic Autoencoder

Let's start with a basic autoencoder that takes a vector of size `n` and compresses it into a vector of size `m`. We will use a fully connected neural network for both the encoder and the decoder.

First, we need to define the encoder and the decoder networks.

``` python
import tensorflow as tf

class AutoEncoder(tf.keras.Model): # keras model sub-classing api
    def __init__(self, inputs_shape):
        super(AutoEncoder, self).__init__()
        self.encoder = self.build_encoder(inputs_shape) # init encoder
        repr_shape = self.encoder.output_shape[1:] # encoded image shape
        self.decoder = self.build_decoder(repr_shape) # init decoder
        
    def build_encoder(self, inputs_shape): # define the encoder model
        inputs = tf.keras.Input(inputs_shape)
        x = tf.keras.layers.Flatten()(inputs) # flatten the input, so we can use dense layers (28*28*1=784)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        encoded = tf.keras.layers.Dense(32, activation='relu')(x)
        return tf.keras.Model(inputs=inputs, outputs=encoded)
    
    def build_decoder(self, repr_shape): # define the decoder model
        inputs = tf.keras.Input(repr_shape)
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(784, activation='sigmoid')(x)
        decoded = tf.keras.layers.Reshape((28,28,1))(x)
        return tf.keras.Model(inputs=inputs, outputs=decoded)
    
    def call(self, inputs): # but them together
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x
```

Here, we have defined the encoder network with three fully connected layers, each with a ReLU activation function. The output of the encoder is a vector of size `repr_shape`. Similarly, we have defined the decoder network with three fully connected layers, again with ReLU activation functions. The output of the decoder a reconstructed image of the same size as the input image.

Now, let's instantiate the autoencoder model.

``` python
# autoencoder model
autoencoder = AutoEncoder((28, 28, 1)) # shape of mnist images (28,28,1)
```

Now, we can train the autoencoder model on a dataset of input vectors.

``` python
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train / 255.
x_test = x_test / 255.

# Compile and train model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))
```

Here, we have loaded the MNIST dataset and preprocessed the data by normalizing the pixel values to be between 0 and 1. We have then compiled the autoencoder model with the binary crossentropy loss function and the Adam optimizer. Finally, we have trained the model on the training data for 10 epochs with a batch size of 256.

Now, we can use the trained autoencoder to compress and decompress input vectors.

``` python
from matplotlib import pyplot as plt

# Get a sample image
sample = x_test[0]
encoded = autoencoder.encoder.predict(sample)
decoded = autoencoder.decoder.predict(encoded)

# Plot the sample image, encoded image and the reconstructed image
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(sample, cmap="gray")
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(encoded, cmap="gray")
plt.title('Encoded Image')
plt.subplot(1, 3, 3)
plt.imshow(decoded, cmap="gray")
```
output:
<img src = "https://i.imgur.com/jqpEVVF.png">

Here, we have taken a sample image from the test set and used the trained autoencoder to compress and decompress it. We can see that the reconstructed image is very similar to the original image.

#### AutoEncoder as a denoiser

autoencoders can also be used for denoising images. We can add some noise to our sample image and then use the autoencoder to reconstruct the original image.

``` python
import numpy as np

# Add noise to the sample image
noise_factor = 0.15
sample_noisy = sample + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=sample.shape)
sample_noisy = np.clip(sample_noisy, 0., 1.)

# Denoise the noisy image
denoised = autoencoder.predict(sample_noisy.reshape((-1,28,28,1)))
denoised = denoised.reshape((28,28,1))

# Plot the noisy image and the reconstructed image
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(sample_noisy, cmap="gray")
plt.title('Noisy Image')
plt.subplot(1, 2, 2)
plt.imshow(denoised, cmap="gray")
plt.title('Denoised Image')
```
output:
<img src = "https://i.imgur.com/eX7E9RE.png">

Here, we have added some noise to the sample image and then used the autoencoder to reconstruct the original image. We can see that the autoencoder has done a good job of removing the noise from the image even though it has never seen any noisy images during training.


## Conclusion

In this blog post, we have explored the basics of autoencoders and implemented them using TensorFlow. We have also seen how autoencoders can be used for dimensionality reduction and for denoising images.