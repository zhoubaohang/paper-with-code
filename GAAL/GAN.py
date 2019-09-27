import tensorflow as tf
import tensorflow.contrib.layers as tcl
from base_model import Model

def sigmoid_cross_entropy_loss(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

class GAN(Model):

    def __init__(self, config_path):
        self.G_name_scope = 'generator'
        self.D_name_scope = 'discriminator'

        self.config = self._load_config(config_path)
    
    def _block_run(self, x, name_scope, reuse=False):
        return self._parse_tf_layers(x, name_scope, self.config[name_scope], reuse=reuse)
    
    def generator(self, z, reuse=False):
        return self._block_run(z, self.G_name_scope, reuse=reuse)
    
    def discriminator(self, x, reuse=False):
        return self._block_run(x, self.D_name_scope, reuse=reuse)

    @property
    def vars(self):
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if var.name.startswith(self.G_name_scope)]
        d_vars = [var for var in t_vars if var.name.startswith(self.D_name_scope)]
        return g_vars + d_vars
    
    def __call__(self, x, z):
        fake_data = self.generator(z)

        real_logits = self.discriminator(x)
        D_prob = tf.nn.sigmoid(real_logits)
        fake_logits = self.discriminator(fake_data, reuse=True)

        D_loss = sigmoid_cross_entropy_loss(tf.zeros_like(fake_logits), fake_logits)\
               + sigmoid_cross_entropy_loss(tf.ones_like(real_logits), real_logits)
        G_loss = sigmoid_cross_entropy_loss(tf.ones_like(fake_logits), fake_logits)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith(self.D_name_scope)]
        g_vars = [var for var in t_vars if var.name.startswith(self.G_name_scope)]
        D_solver = tf.train.AdamOptimizer(1e-4).minimize(D_loss, var_list=d_vars)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars)

        return {
            'D_loss': D_loss,
            'D_solver': D_solver,
            'G_loss': G_loss,
            'G_solver': G_solver,
            'fake_data': fake_data,
            'D_prob': D_prob
        }