import tensorflow as tf
import pdb

model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

with open('model_summary.txt', 'w') as f:
	for layer in model.layers:
		f.write(layer.name + '\t' + str(layer.output_shape) + '\n')