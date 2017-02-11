import numpy as np
import os
import csv
import cv2
import random

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D

import sklearn
from sklearn.model_selection import train_test_split

class Sample(object):
  def __init__(self, image, steering):
    self.image_ = image
    self.steering_ = steering

data_directory = 'data/'

# First, read the sample images for training.
samples = []
with open(data_directory + 'driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    # Skip header line
    if line[0] == 'center':
      continue
    center_image = line[0].split('/')[-1]
    samples.append(Sample(
      os.path.join(data_directory, 'IMG', center_image),
      float(line[3])
    ))
print('Total # samples: ', len(samples))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
  """A generator that shuffles and reads appropriate images.

  Args:
    samples: List of Sample objects.
    batch_size: The number of data examples per batch.
  """
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    random.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset : offset + batch_size]

      images = []
      angles = []
      for batch_sample in batch_samples:
        images.append(cv2.imread(batch_sample.image_))
        angles.append(batch_sample.steering_)

      # trim image to only see section with road.
      X_train = np.array(images)[:, 80:, :, :]
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320

model = Sequential()
# Normalize the image to be between (-1, 1).
model.add(Lambda(lambda x: (x - 127.5) / 127.5, input_shape=(row, col, ch)))

# Model based on LeNet architecture.
model.add(Convolution2D(nb_filter=6, nb_row=5, nb_col=5, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(nb_filter=16, nb_row=5, nb_col=5, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(
  train_generator,
  samples_per_epoch=len(train_samples),
  validation_data=validation_generator,
  nb_val_samples=len(validation_samples),
  nb_epoch=3)

model.save('model.h5')
