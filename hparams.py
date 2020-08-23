
class Hparams:
	def __init__(self):
		### data and save path
		self.train_record_path = '/home/tuanluong/gpu/Documents/code/id_number/id_number.valid'
		# self.train_record_path = '/Users/admin/Documents/code/id_number/temp.valid'
		self.num_train_sample = 3213
		self.valid_record_path = '/home/tuanluong/gpu/Documents/code/id_number/id_number.valid'
		# self.valid_record_path = '/Users/admin/Documents/code/id_number/temp.valid'
		self.charset_path = 'charsets/charset_size=11.txt'
		self.num_valid_sample = 3213
		self.save_path = 'training_checkpoints'
		self.save_best = False
		self.max_to_keep = 1000

		### input params
		self.image_shape = (64, 500, 3)
		self.nul_code = 10
		self.charset_size = 11
		self.max_char_length = 13

		### conv_tower params
		# base model from tf.keras.application, or custom instance of tf.keras.Model
		# check for new models from https://www.tensorflow.org/api_docs/python/tf/keras/applications
		# check for newest model from tf-nightly version
		self.base_model_name = 'InceptionResNetV2'
		# last convolution layer from base model which extract features from
		# inception v3: mixed2 (mixed_5d in tf.slim inceptionv3)
		# inception resnet v2: (mixed_6a in tf.slim inception_resnet_v2)
		self.end_point = 'mixed_6a'
		# endcode cordinate feature to conv_feature
		self.use_encode_cordinate = True

		### RNN tower params
		self.rnn_cell = 'lstm'
		self.rnn_units = 256
		self.dense_units = 256
		self.weight_decay = 0.00004

		### attention params

		### training params
		self.batch_size = 32
		self.max_epochs = 1000
		self.lr = 0.001

hparams = Hparams()