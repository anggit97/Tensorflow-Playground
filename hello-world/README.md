# Define And Compile Neural Network
make model using 1 layer, 1 neuron, and 1 input shape

``
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
``

calculate loss and optimized function

``
model.compile(optimizer='sgd', loss='mean_squared_error')
``

# Providing Data 
``
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
``

# Training
``
model.fit(xs, ys, epochs=50)
``

result : 
``
6/6 [==============================] - 0s 324us/sample - loss: 0.1182
Epoch 30/50
6/6 [==============================] - 0s 464us/sample - loss: 0.0955
Epoch 31/50
6/6 [==============================] - 0s 505us/sample - loss: 0.0776
Epoch 32/50
6/6 [==============================] - 0s 359us/sample - loss: 0.0635
Epoch 33/50
6/6 [==============================] - 0s 607us/sample - loss: 0.0523
Epoch 34/50
6/6 [==============================] - 0s 497us/sample - loss: 0.0435
Epoch 35/50
6/6 [==============================] - 0s 522us/sample - loss: 0.0365
Epoch 36/50
6/6 [==============================] - 0s 409us/sample - loss: 0.0309
Epoch 37/50
6/6 [==============================] - 0s 309us/sample - loss: 0.0265
``

# Test and Using the model
``
print(model.predict([4.0]))
``

result

``
[[13.067166]]
``

source : https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/index.html?index=..%2F..index#0