import numpy as np
from mnist_loader import *
from base_model import Model
from tqdm import trange
import matplotlib.pyplot as plt

class MNIST(Model):

    def __init__(self, config_path):
        super().__init__()
        self.logger.info('Loading data from MNIST.......')
        self.config = self._load_config(config_path)
        self.data, self.label = self.__collect_data(load_data())
        self.__split_data()

    def __collect_data(self, data):
        self.selected_numbers = self.config['select']
        xs, ys = [], []

        for x, y in data:
            for num in self.selected_numbers:
                selected_idx = np.where(y == num)
                xs.append(x[selected_idx])
                label = np.zeros([len(selected_idx[0]), len(self.selected_numbers)])
                label[:, self.selected_numbers.index(num)] = 1.
                ys.append(label)
                
        return (np.vstack(xs), np.vstack(ys))
    
    def __split_data(self):
        idx = [i for i in range(self.data.shape[0])]
        split = self.config['split']
        assert sum(split.values()) == 1., 'The division of dataset is illegal!'
        np.random.shuffle(idx)

        startIdx = 0
        self.split_data = {}
        for k, v in split.items():
            endIdx = startIdx + int(v * len(idx))
            self.split_data[k] = (self.data[idx[startIdx:endIdx]],\
                                  self.label[idx[startIdx:endIdx]])
            startIdx = endIdx

    def parseLabel(self, labels):
        onehot_labels = np.zeros((len(labels), len(self.selected_numbers)))
        for i in range(len(labels)):
            idx = self.selected_numbers.index(labels[i])
            onehot_labels[i, idx] = 1.
        return onehot_labels
    
    def addData(self, data, labels):
        oldData, oldLables = self.split_data['label']
        data = np.vstack([oldData, data])
        labels = np.vstack([oldLables, labels])
        self.split_data['label'] = (data, labels)
    
    def removeData(self, idx):
         oldData, oldLables = self.split_data['unlabel']
         data = np.delete(oldData, idx, axis=0)
         labels = np.delete(oldLables, idx, axis=0)
         self.split_data['unlabel'] = (data, labels)
    
    def getData(self, mode='all'):
        if mode == 'all':
            data, label = [], []
            for v in self.split_data.values():
                data.append(v[0])
                label.append(v[1])
            return (np.vstack(data), np.vstack(label))
        else:
            return self.split_data[mode]
    
    def next_batch(self, mode='all', batch_size=1):
        data, label = [], []
        if mode == 'all':
            for v in self.split_data.values():
                data.append(v[0])
                label.append(v[1])
            data, label = (np.vstack(data), np.vstack(label))
        else:
            data, label = self.split_data[mode]
        idx = [i for i in range(len(data))]
        np.random.shuffle(idx)
        idx = np.array(idx)

        for i in trange(len(idx)//batch_size, ascii=True):
            x = data[idx[i*batch_size:(i+1)*batch_size]]
            y = label[idx[i*batch_size:(i+1)*batch_size]]
            yield (x, y)
        
if __name__ == "__main__":
    data_loader = MNIST('config.yaml')

    from pyemd import emd_samples

    label_data, _  = data_loader.getData(mode='label')
    unlabel_data, _ = data_loader.getData(mode='unlabel')
    label_data = np.vstack((label_data, unlabel_data[0]))
    all_data, _ = data_loader.getData()
    print(emd_samples(label_data.flatten(), all_data[:2000].flatten()))
    # for x, y in data_loader.next_batch(batch_size=64):
    #     print(y)
    # plt.imshow(data_loader.getData('label')[0][i].reshape((28,28)), cmap='gray')
    # plt.show()