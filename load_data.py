import pickle as pk
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def download_and_extract_data(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                                float(count * block_size) / float(total_size) * 100.0
                                                                )
                                )
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)

def unpickle(file):
    fo = open(file, 'rb')
    d = pk.load(fo,encoding='iso-8859-1')
    fo.close()
    return {'x': np.cast[np.float32](
                                        (-127.5 
                                        + d['data'].reshape((10000,
                                                                3,
                                                                32,
                                                                32))
                                        )
                                    /128.
                                    ), 
            'y': np.array(d['labels']).astype(np.uint8)
            }

def load(data_dir, subset, download=True):
    if download:
        download_and_extract_data(data_dir)

    if subset=='train':
        train_data = [unpickle(os.path.join(data_dir,'cifar-10-batches-py/data_batch_' + str(i))) for i in range(1,6)]
        x_train = np.concatenate([batch['x'] for batch in train_data],axis=0)
        y_train = np.concatenate([batch['y'] for batch in train_data],axis=0)
        return x_train[:1000], y_train[:1000]

    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'cifar-10-batches-py/test_batch'))
        x_test = test_data['x']
        y_test = test_data['y']
        return x_test[:100], y_test[:100]

    else:
        raise NotImplementedError('subset can only be train or test')
