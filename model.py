import argparse
import csv
import os
import random

import numpy as np
import cv2

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D

import sklearn
from sklearn.model_selection import train_test_split

class Sample(object):
  def __init__(self, image, steering, flip):
    self.image_ = image
    self.steering_ = steering
    self.flip_ = flip

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument(
    'model',
    type=str,
    help='Path to store model h5 file. Model should be on the same path.'
)
parser.add_argument(
    'steering_adjustment',
    type=float,
    default=0.2,
    nargs='?',
    help='Steering adjustment to the left / right camera.'
)
parser.add_argument(
    'validation_size',
    type=float,
    default=0.3,
    nargs='?',
    help='Ratio of data to be used for validation.'
)
parser.add_argument(
    'num_epochs',
    type=int,
    default=3,
    nargs='?',
    help='# epochs to be used for training.'
)
args = parser.parse_args()

# First, read the sample images for training.
samples = []
def ReadSamples(directory, shift=False):
  results = []
  with open(os.path.join(directory, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      # Skip header line
      if line[0] == 'center':
        continue

      # Skip those with steering angle 0, which is most likely noise during
      # data collection since the steering is done using keystroke and not
      # continuous.
      steering = float(line[3])
      if steering == 0:
        continue

      center_image = line[0].split('/')[-1]
      results.append(Sample(
          os.path.join(directory, 'IMG', center_image),
          steering,
          flip=False,
      ))
        # Generate another mirror image by flipping the image
        # and steering angle so that the generator can sample from
        # both images appropriately.
      results.append(Sample(
          os.path.join(directory, 'IMG', center_image),
          steering,
          flip=True
      ))
      if shift:
        # Use the left / right image and their flipped version.
        results.append(Sample(
            os.path.join(directory, 'IMG', line[1].split('/')[-1]),
            steering + args.steering_adjustment,
            flip=False
        ))
        results.append(Sample(
            os.path.join(directory, 'IMG', line[1].split('/')[-1]),
            steering + args.steering_adjustment,
            flip=True
        ))
        results.append(Sample(
            os.path.join(directory, 'IMG', line[2].split('/')[-1]),
            steering - args.steering_adjustment,
            flip=False
        ))
        results.append(Sample(
            os.path.join(directory, 'IMG', line[2].split('/')[-1]),
            steering - args.steering_adjustment,
            flip=True
        ))
  return results

samples = []
# This is Udacity provided data.
samples.extend(ReadSamples('data', shift=True))
# This is a directory with a bunch of recovery data.
samples.extend(ReadSamples('more_data', shift=False))
# This is a directory with a bunch of recovery data for the first curve after the bridge.
samples.extend(ReadSamples('more_cheat', shift=False))
# This is a directory with a bunch of recovery data for the second curve after the bridge.
samples.extend(ReadSamples('next_cheat', shift=False))
print('Total # samples: ', len(samples))

train_samples, validation_samples = train_test_split(samples, test_size=args.validation_size)

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
        if not batch_sample.flip_:
          images.append(cv2.resize(cv2.imread(batch_sample.image_), None, fx=0.5, fy=0.5))
          angles.append(batch_sample.steering_)
        else:
          images.append(cv2.resize(np.fliplr(cv2.imread(batch_sample.image_)), None, fx=0.5, fy=0.5))
          angles.append(-batch_sample.steering_)
      # trim image to only see section with road.
      X_train = np.array(images)[:, 30:-10, :, :]
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 40, 160

model = Sequential()

# Normalize the image to be between (-1, 1).
model.add(Lambda(lambda x: (x - 127.5) / 127.5, input_shape=(row, col, ch)))

# Model based on LeNet architecture with some modification.
model.add(Convolution2D(nb_filter=10, nb_row=3, nb_col=3, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(nb_filter=24, nb_row=3, nb_col=3, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(
  train_generator,
  samples_per_epoch=len(train_samples),
  validation_data=validation_generator,
  nb_val_samples=len(validation_samples),
  nb_epoch=args.num_epochs)

model.save(args.model)
