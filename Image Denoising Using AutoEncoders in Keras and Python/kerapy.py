import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import random
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(plt.imshow(X_train[0], cmap = 'gray'))
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

i = random.randint(1, 60000)
print(plt.imshow(X_train[i], cmap='gray'))
label = y_train[i]
print(label)

# Let's view more images in a grid format
# Define the dimensions of the plot grid
W_grid = 15
L_grid = 15

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_training = len(X_train) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid):
    index = np.random.randint(0, n_training)
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index], fontsize=8)
    axes[i].axis('off')

X_train = X_train/255
X_test = X_test/255
print(X_train)

noise_factor = 0.3
noise_dataset = []
for img in X_train:
    noisy_image = img + noise_factor * np.random.randn(*img.shape)
    noise_image = np.clip(noisy_image, 0, 1)
    noise_dataset.append(noisy_image)

noise_factor = 0.1
noise_test_dataset = []
for img in X_test:
    noisy_image = img + noise_factor * np.random.randn(*img.shape)
    noisy_image = np.clip(noisy_image, 0, 1)
    noise_test_dataset.append(noisy_image)

noise_dataset = np.array(noise_dataset)
noise_test_dataset = np.array(noise_test_dataset)

print(plt.imshow(noise_dataset[22], cmap='gray'))

autoencoder = tf.keras.models.Sequential()
autoencoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', input_shape=(28, 28, 1)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='sigmoid'))
autoencoder.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
autoencoder.summary()

autoencoder.fit(noise_dataset.reshape(-1, 28, 28, 1),
               X_train.reshape(-1, 28, 28, 1),
               epochs =10,
               batch_size =200,
               validation_data=(noise_test_dataset.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)))

evaluation = autoencoder.evaluate(noise_test_dataset.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1))
print('Test loss : {:.3f}'.format(evaluation))

predicted = autoencoder.predict(noise_test_dataset[:10].reshape(-1, 28, 28, 1))

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([noise_test_dataset[:10], predicted], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
