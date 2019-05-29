from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import normc_initializer

import tensorflow as tf


class ConvModelGlobalObs(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.
        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.
        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].
        When using dict or tuple observation spaces, you can access
        the nested sub-observation batches here as well:
        Examples:
            >>> print(input_dict)
            {'prev_actions': <tf.Tensor shape=(?,) dtype=int64>,
             'prev_rewards': <tf.Tensor shape=(?,) dtype=float32>,
             'is_training': <tf.Tensor shape=(), dtype=bool>,
             'obs': (observation, features)
        """
        # Convolutional Layer #1

        Relu = tf.nn.relu
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.contrib.layers.fully_connected

        map_size = int(input_dict['obs'][0].shape[0])

        N_CHANNELS = 96

        conv1 = Relu(self.conv2d(input_dict['obs'], N_CHANNELS, 'valid', strides=(2, 2)))

        # conv2 = Relu(self.conv2d(conv1, 64, 'valid'))

        # conv3 = Relu(self.conv2d(conv2, 64, 'valid'))

        conv2_flat = tf.reshape(conv1, [-1, int(N_CHANNELS * ((map_size-3 + 1)/2)**2)])
        # conv4_feature = tf.concat((conv2_flat, input_dict['obs'][1]), axis=1)
        s_fc1 = Relu(Dense(conv2_flat, 256))
        layerN_minus_1 = Relu(Dense(s_fc1, 64))
        layerN = Dense(layerN_minus_1, num_outputs)
        return layerN, layerN_minus_1

    def conv2d(self, x, out_channels, padding, strides=(1,1)):
        return tf.layers.conv2d(x, out_channels, kernel_size=[3, 3], padding=padding,
                                use_bias=True, strides=strides)


class LightModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.
        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.
        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].
        When using dict or tuple observation spaces, you can access
        the nested sub-observation batches here as well:
        Examples:
            >>> print(input_dict)
            {'prev_actions': <tf.Tensor shape=(?,) dtype=int64>,
             'prev_rewards': <tf.Tensor shape=(?,) dtype=float32>,
             'is_training': <tf.Tensor shape=(), dtype=bool>,
             'obs': (observation, features)
        """
        # print(input_dict)
        # Convolutional Layer #1
        self.sess = tf.get_default_session()
        Relu = tf.nn.relu
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.contrib.layers.fully_connected

        #conv1 = Relu(self.conv2d(input_dict['obs'][0], 32, 'valid'))
        conv1 = Relu(self.conv2d(input_dict['obs'], 32, 'valid'))
        conv2 = Relu(self.conv2d(conv1, 16, 'valid'))

        # conv3 = Relu(self.conv2d(conv2, 64, 'valid'))

        conv4_flat = tf.reshape(conv2, [-1, 16 * (17-2*2)**2])
        #conv4_feature = tf.concat((conv4_flat, input_dict['obs'][1]), axis=1)
        s_fc1 = Relu(Dense(conv4_flat, 128, weights_initializer=normc_initializer(1.0)))
        # layerN_minus_1 = Relu(Dense(s_fc1, 256, use_bias=False))
        layerN = Dense(s_fc1, num_outputs, weights_initializer=normc_initializer(0.01))
        return layerN, s_fc1

    def conv2d(self, x, out_channels, padding):
        return tf.layers.conv2d(x, out_channels, kernel_size=[3, 3], padding=padding, use_bias=True)
                                # weights_initializer=normc_initializer(1.0))
