import tensorflow as tf
import matplotlib.pyplot as plt

#Prepare dataset
mnist = tf.keras.datasets.mnist

#Load dataset
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# plt.imshow(training_images[10])
# print(training_labels[10])
# print(training_images[10])

#Normlization => convert it become 0 or 1 for each array item
training_images = training_images / 255.0
test_images = test_images / 255.0

#Design the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)

#Compiling and Training
model.compile(
    optimizer = tf.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(training_images, training_labels, epochs=5)

#Test the model
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])