import os
import yaml
import logging
import tensorflow as tf
import tensorflow.contrib.layers as tcl

tf.set_random_seed(1)

class Model(object):

    def __init__(self):
        class_name = self.__class__.__name__
        self.logger = logging.getLogger(class_name)
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=logging.INFO)

    def _load_config(self, config_path):
        assert os.path.exists(config_path), 'configuration file [%s] not found' % config_path

        config = None
        with open(config_path, 'r') as fp:
            config = yaml.load(fp.read())[self.__class__.__name__]
        
        return config
    
    def _parse_tf_layers(self, x, name_scope, layers, reuse=False):
        hiddens = [x]
        with tf.variable_scope(name_scope, reuse=reuse):
            for k, v in layers.items():
                with tf.variable_scope(k):
                    size = v['size']
                    active_fn = eval(v['active_fn']) if v['active_fn'] else None
                    hidden = tcl.fully_connected(hiddens[-1], v['size'], activation_fn=active_fn,
                                                 weights_initializer=tf.random_normal_initializer(0, 0.02))
                    hiddens.append(hidden)
        
        return hiddens[-1]