import numpy as np
import tensorflow as tf
from tensorflow import keras

#Create model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#Prepare datasets
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

#training
model.fit(xs, ys, epochs=50)

print(model.predict([4.0]))