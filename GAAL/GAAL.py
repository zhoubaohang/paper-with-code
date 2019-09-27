import os
import yaml
import numpy as np
from GAN import GAN
import tensorflow as tf
from base_model import Model
from data_loader import MNIST
from pyemd import emd_samples
from Classifier import Classifier
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from losses import maximum_mean_discrepancy
from tqdm import trange

class GAAL(Model):

    def __init__(self, config_path='config.yaml'):
        super().__init__()
        config = self._load_config(config_path)
        self.AL_epoch = config['AL_epoch']
        self.GAN_epoch = config['GAN_epoch']
        self.input_dim = config['input_dim']
        self.query_num = config['query_num']
        self.output_dim = config['output_dim']
        self.latent_dim = config['latent_dim']
        self.batch_size = config['batch_size']
        self.model_path = config['model_path']
        self.model_name = config['model_name']
        self.lambd = config['lambda']
        self.index = 0.5

        # init GAN model
        self.gan = GAN(config_path)
        # init Classifier model
        self.classifier = Classifier(config_path)
        # init data loader
        self.data_loader = MNIST(config_path)
        # init GAAL's tensorflow parameters
        self.__init_parameters()

    def __init_parameters(self):
        self.tf_x = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        self.tf_source_data = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        self.tf_y = tf.placeholder(tf.float32, shape=[None, self.output_dim])
        self.tf_z = tf.placeholder(tf.float32, shape=[None, self.latent_dim])
        self.trainable_z = tf.get_variable('trainable_latent_value', shape=[self.query_num, self.latent_dim],
                                           initializer=tf.random_normal_initializer(mean=0,stddev=1))

        self.distribute_distance = maximum_mean_discrepancy(self.tf_x, self.tf_source_data)

        self.GAN_params = self.gan(self.tf_x, self.tf_z)
        trainable_fake_data = self.gan.generator(self.trainable_z, reuse=True)
        self.Classifier_params = self.classifier(self.tf_x, self.tf_y, trainable_fake_data, \
                                                 self.trainable_z, self.tf_source_data)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
    
    def trainGAN(self):
        self.logger.info('Begin to train GAN.......')
        cnt = 0
        full_path = '{}{}'.format(self.model_path, self.model_name)

        # if os.path.exists('{}.meta'.format(full_path)):
        #     self.saver.restore(self.sess, full_path)
        # else:
        for _ in range(self.GAN_epoch):
            
            for x, _ in self.data_loader.next_batch(mode='label', batch_size=self.batch_size):
                cnt += 1
                feed_dict = {
                    self.tf_x: x,
                    self.tf_z: np.random.normal(size=[self.batch_size, self.latent_dim])*self.lambd
                }

                for _ in range(1):
                    self.sess.run(self.GAN_params['G_solver'], feed_dict=feed_dict)

                self.sess.run(self.GAN_params['D_solver'], feed_dict=feed_dict)

                # G_loss, D_loss = self.sess.run([self.GAN_params['G_loss'], self.GAN_params['D_loss']], feed_dict=feed_dict)
                # print('G_loss {} D_loss {}'.format('%.4f'%G_loss, '%.4f'%D_loss))
            
                # if cnt % 10000 == 0:
                #     fake_data = self.sess.run(self.GAN_params['fake_data'], feed_dict={
                #         self.tf_z: np.random.normal(size=[self.batch_size, self.latent_dim])
                #     })
                #     fig = plt.figure()
                #     for i in range(10):
                #         ax = fig.add_subplot(1, 10, i+1)
                #         ax.imshow(fake_data[i].reshape(28, 28), cmap='gray')
                #         ax.axis('off')
                #     fig.savefig('iteration%d' % cnt)
                #     plt.close(fig)
            # self.saver.save(self.sess, full_path)
        
    def trainAL(self, mode='entropy'):
        self.logger.info('Begin to train AL.......')
        accs = []
        samples = []

        for e in range(self.AL_epoch):
            
            if mode == 'distribute':
                # reset Classifier parameters
                # self.sess.run(tf.variables_initializer(self.gan.vars))
                self.trainGAN()

            # reset Classifier parameters
            self.sess.run(tf.variables_initializer(self.classifier.vars))

            # classifier train procedure
            tmp = []

            for i in range(10):
                self.logger.info('Begin to train Classifier epoch[{}].......'.format(i+1))
                for x, y in self.data_loader.next_batch('label', batch_size=32):
                    feed_dict = {
                        self.tf_x: x,
                        self.tf_y: y
                    }
                    self.sess.run(self.Classifier_params['clf_solver'], feed_dict=feed_dict)
                    
                test_data, test_label = self.data_loader.getData(mode='test')
                feed_dict = {
                    self.tf_x: test_data,
                    self.tf_y: test_label
                }
                acc = self.sess.run(self.Classifier_params['accuracy'], feed_dict=feed_dict)
                tmp.append(acc)
            accs.append(max(tmp))

            # active learning procedure
            unlabel_data, unlabel_label = self.data_loader.getData('unlabel')
            feed_dict = {
                self.tf_x: unlabel_data
            }
            unlabel_D_prob = self.sess.run(self.GAN_params['D_prob'], feed_dict=feed_dict)
            unlabel_D_prob = unlabel_D_prob.flatten()
            # entropy select
            unlabel_prob = self.sess.run(self.Classifier_params['prob'], feed_dict=feed_dict)
            entropy = -np.sum(np.log(unlabel_prob) * unlabel_prob, axis=1)

            if mode == 'entropy':
                idxs = np.argsort(entropy)[-self.query_num:]

            if mode == 'distribute':
                score = entropy**(1. + self.index**e)*(1. - unlabel_D_prob)**(1. - self.index**e)
                idxs = np.argsort(score)[-self.query_num:]

            if mode == 'random':
                # random select
                idxs = np.random.choice(unlabel_label.shape[0], size=self.query_num, replace=False).tolist()

            data = unlabel_data[np.array(idxs)]
            label = unlabel_label[np.array(idxs)]
            self.data_loader.removeData(idxs)
            self.data_loader.addData(data, label)
            samples.append(samples[-1]+len(idxs) if len(samples) \
                            else self.data_loader.getData('label')[1].shape[0])

            # self.sess.run(tf.variables_initializer([self.trainable_z]))
            # test_data, test_label = self.data_loader.getData(mode='test')
            # for x, _ in self.data_loader.next_batch(mode='unlabel', batch_size=self.batch_size):
            #     feed_dict = {
            #         self.tf_source_data: x,
            #         self.tf_x: test_data,
            #         self.tf_y: test_label
            #     }
            #     self.sess.run([self.Classifier_params['latent_solver']], feed_dict=feed_dict)
            # optimized_z = self.sess.run(self.trainable_z)
            # generated_data = self.sess.run(self.GAN_params['fake_data'], feed_dict={self.tf_z:optimized_z})
            # labels = []
            # for i in range(len(generated_data)):
            #     plt.imshow(generated_data[i].reshape((28,28)), cmap='gray')
            #     plt.axis('off')
            #     plt.show()
            #     labels.append(int(input('Please input the number as you see:')))
            # onehot_labels = self.data_loader.parseLabel(labels)
            # self.data_loader.addData(generated_data, onehot_labels)
            
        print(max(accs))
        test_data, test_label = self.data_loader.getData(mode='test')
        unlabel_data, unlabel_label = self.data_loader.getData(mode='unlabel')
        test_data = np.vstack((test_data, unlabel_data))
        test_label = np.vstack((test_label, unlabel_label))
        feed_dict = {
            self.tf_x: test_data,
            self.tf_y: test_label
        }
        acc = self.sess.run(self.Classifier_params['accuracy'], feed_dict=feed_dict)
        print(acc)
        plt.axis([0, max(samples), 0.4, 1.])

        label = mode
        if mode == 'distribute':
            label = "{}-{}-{}".format(label, '%.1f'%self.lambd, '%.1f'%self.index)
        plt.plot(samples, accs,  '.-', label=label)
        

if __name__ == "__main__":
    model = GAAL()
    model.trainAL('entropy')

    tf.reset_default_graph()
    model = GAAL()
    model.trainAL('random')

    tf.reset_default_graph()
    model = GAAL()
    model.lambd = 0.9
    model.index = 0.7
    model.trainAL('distribute')
 
    plt.legend(loc='best')
    plt.show()


