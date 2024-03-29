from custom_keras_layers import *
from datetime import datetime
from dataset import Dataset
from hparams import hparams
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import pdb
import os

class Model(object):
    def __init__(self):
        self.char_mapping = {}
        with open(hparams.charset_path, 'r') as f:
            for row in f:
                index, label = row[:-1].split('\t')
                self.char_mapping[int(index)] = label

    def loss_function(self, real, pred):
        loss_ = self.loss_object(real, pred)
        return tf.reduce_mean(loss_)

    def create_model(self):
        ### create model
        self.best_val_acc = 0.0
        # dataset
        self.train_dataset = Dataset(hparams, hparams.train_record_path)
        self.valid_dataset = Dataset(hparams, hparams.valid_record_path)
        self.train_dataset.load_tfrecord()
        self.valid_dataset.load_tfrecord()
        # create conv layer
        self.conv_layer = ConvBaseLayer(hparams)
        # create attention + RNN layer
        self.decoder = Decoder(hparams)
        
        ### define training ops and params
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.lr)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        self.last_epoch = 0
        self.train_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/train')
        self.valid_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/valid')
        self.checkpoint_dir = os.path.join(hparams.save_path, 'train')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.conv_layer,
                                              decoder=self.decoder)

    def load_model(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest != None:
            logging.info('load model from {}'.format(latest))
            self.last_epoch = int(latest.split('-')[-1])
            self.checkpoint.restore(latest)

    def train_step(self, batch_input, batch_target):
        loss = 0
        current_batch_size = batch_input.shape[0]
        with tf.GradientTape() as tape:
            conv_out = self.conv_layer(batch_input)
            dec_input = tf.cast(tf.expand_dims([hparams.charset_size] * current_batch_size, 1), tf.float32)
            dec_hidden = self.decoder.initial_hidden_state(current_batch_size)

            for t in range(batch_target.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, conv_out)
                loss += self.loss_function(batch_target[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(batch_target[:, t], 1)
        variables = self.conv_layer.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return (loss / int(batch_target.shape[1]))

    def evaluate(self, batch_input, batch_target):
        current_batch_size = batch_input.shape[0]
        conv_out = self.conv_layer(batch_input)
        dec_input = tf.cast(tf.expand_dims([hparams.charset_size] * current_batch_size, 1), tf.float32)
        dec_hidden = self.decoder.initial_hidden_state(current_batch_size)

        batch_true_char = 0
        batch_true_str  = np.ones(current_batch_size)
        batch_predictions = np.zeros((current_batch_size, hparams.max_char_length))

        for t in range(batch_target.shape[1]):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, conv_out)
            # storing the attention weights to plot later on
            dec_input = tf.argmax(predictions, axis=1)
            predictions = dec_input.numpy()
            dec_input = tf.cast(tf.expand_dims(dec_input, axis=1), tf.float32)
            batch_predictions[:, t] = predictions
            batch_true_char += np.sum(predictions == batch_target[:, t])
            batch_true_str = np.logical_and(batch_true_str, (predictions == batch_target[:, t]))
        batch_true_str = np.sum(batch_true_str)
        return batch_true_char, batch_true_str


    def inference(self, image):
        current_batch_size = 1
        conv_out = self.conv_layer(np.array([image]))
        dec_input = tf.cast(tf.expand_dims([hparams.charset_size] * current_batch_size, 1), tf.float32)
        dec_hidden = self.decoder.initial_hidden_state(current_batch_size)
        result = ''

        for t in range(hparams.max_char_length):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, conv_out)
            # storing the attention weights to plot later on
            dec_input = tf.argmax(predictions, axis=1)
            char_index = dec_input.numpy()[0]
            result += self.char_mapping[char_index]
            dec_input = tf.cast(tf.expand_dims(dec_input, axis=1), tf.float32)
            
        return result


    def train(self):
        self.load_model()
        for epoch in range(self.last_epoch, hparams.max_epochs):
            start = datetime.now()
            total_loss = 0

            # train each batch in dataset
            for batch, (batch_input, batch_target) in enumerate(self.train_dataset.dataset):
                batch_loss = self.train_step(batch_input, batch_target)
                total_loss += batch_loss
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

            # evaluate on train set
            cnt_true_char = 0
            cnt_true_str = 0
            sum_char = 0
            sum_str = 0
            for batch, (batch_input, batch_target) in enumerate(self.train_dataset.dataset):
                batch_true_char, batch_true_str = self.evaluate(batch_input, batch_target)
                cnt_true_char += batch_true_char
                cnt_true_str  += batch_true_str
                sum_char += batch_input.shape[0] * hparams.max_char_length
                sum_str  += batch_input.shape[0]
            train_char_acc = cnt_true_char/sum_char
            train_str_acc  = cnt_true_str/sum_str

            # evaluate on train set
            cnt_true_char = 0
            cnt_true_str = 0
            sum_char = 0
            sum_str = 0
            for batch, (batch_input, batch_target) in enumerate(self.valid_dataset.dataset):
                batch_true_char, batch_true_str = self.evaluate(batch_input, batch_target)
                cnt_true_char += batch_true_char
                cnt_true_str  += batch_true_str
                sum_char += batch_input.shape[0] * hparams.max_char_length
                sum_str  += batch_input.shape[0]
            valid_char_acc = cnt_true_char/sum_char
            valid_str_acc  = cnt_true_str/sum_str

            # save checkpoint
            if hparams.save_best:
                if self.best_val_acc < valid_str_acc:
                    self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            else:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            # write log
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', total_loss, step=epoch)
                tf.summary.scalar('character accuracy', train_char_acc, step=epoch)
                tf.summary.scalar('sequence accuracy', train_str_acc, step=epoch)

            with self.valid_summary_writer.as_default():
                tf.summary.scalar('character accuracy', valid_char_acc, step=epoch)
                tf.summary.scalar('sequence accuracy', valid_str_acc, step=epoch)

            # log traing result of each epoch
            logging.info('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batch))
            logging.info('Accuracy on train set:')
            logging.info('character accuracy: {:.6f}'.format(train_char_acc))
            logging.info('sequence accuracy : {:.6f}'.format(train_str_acc))
            logging.info('Accuracy on valid set:')
            logging.info('character accuracy: {:.6f}'.format(valid_char_acc))
            logging.info('sequence accuracy : {:.6f}'.format(valid_str_acc))
            logging.info('Time taken for 1 epoch {} sec\n'.format(datetime.now() - start))
