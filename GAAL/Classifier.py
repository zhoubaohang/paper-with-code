import tensorflow as tf
import tensorflow.contrib.layers as tcl
from base_model import Model
from losses import mmd_loss, maximum_mean_discrepancy

class Classifier(Model):

    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.name_scope = 'Classifier'
    
    @property
    def vars(self):
        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if var.name.startswith(self.name_scope)]
        return c_vars
    
    def __call__(self, x, y, fake_data, z, source_data):
        logits = self._parse_tf_layers(x, self.name_scope, self.config)
        # prediction metrics
        prob = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # classification params
        t_vars = tf.trainable_variables()
        clf_vars = [var for var in t_vars if var.name.startswith(self.name_scope)]
        clf_loss = tf.losses.softmax_cross_entropy(y, logits)
        clf_solver = tf.train.AdamOptimizer(beta1=0.5, beta2=0.9).minimize(clf_loss, var_list=clf_vars)
        
        # optimize latent
        # maxprob = tf.argmax(prob, 1)
        # idx = tf.argsort(maxprob, direction='ASCENDING')
        # num = z.get_shape().as_list()[0]
        # prob = tf.gather(prob, idx)[:num]
        # fake_logits = self._parse_tf_layers(fake_data, self.name_scope, self.config, reuse=True)
        # latent_loss = tf.losses.softmax_cross_entropy(prob, fake_logits)

        fake_logits = self._parse_tf_layers(fake_data, self.name_scope, self.config, reuse=True)
        class_num = fake_logits.get_shape().as_list()[-1]
        latent_loss = mmd_loss(source_data, fake_data, 1.) +\
                      tf.losses.softmax_cross_entropy(tf.ones_like(fake_logits)/class_num, fake_logits)
        latent_solver = tf.train.AdamOptimizer().minimize(latent_loss, var_list=z)

        return {
            'accuracy': accuracy,
            'clf_loss': clf_loss,
            'clf_solver': clf_solver,
            'latent_loss': latent_loss,
            'latent_solver': latent_solver,
            'prob': prob
        }