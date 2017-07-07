import numpy as np
from scipy import misc

class MNISTDataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, path, is_train):
    self.is_train = is_train
    self.path = path
    self.data = self._get_data()

  def _get_data(self):
    from tensorflow.contrib.learn.python.learn.datasets.base \
      import maybe_download
    from tensorflow.contrib.learn.python.learn.datasets.mnist \
      import extract_images, extract_labels

    if self.is_train:
      IMAGES = 'train-images-idx3-ubyte.gz'
      LABELS = 'train-labels-idx1-ubyte.gz'
    else :
      IMAGES = 't10k-images-idx3-ubyte.gz'
      LABELS = 't10k-labels-idx1-ubyte.gz'
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    local_file = maybe_download(IMAGES, self.path, SOURCE_URL)
    with open(local_file, 'rb') as f:
      images = np.empty([2,500,458,1], dtype= np.float)
      image= misc.imread('D:\LEIDEN\Applicator\Binary_mask/bm1_28.jpg')
      image = misc.imresize(image, [500, 458])
      model= misc.imread('D:/LEIDEN/Applicator/model_28.jpg')
      model=misc.imresize(model,[500,458])

      images[0,:,:,0]=image
      images[1, :, :, 0] = model

      #images = extract_images(f)
    local_file = maybe_download(LABELS, self.path, SOURCE_URL)
    with open(local_file, 'rb') as f:
      labels =np.zeros([2],dtype=np.uint8)# extract_labels(f, one_hot=False)

    values, counts = np.unique(labels, return_counts=True)
    #counts=[0,1];
    data = []
    for i in range(1):#10
      label = values[i]
      count = counts[i]
      arr = np.empty([count, 500, 458,1], dtype=np.float32)
      data.append(arr)

    l_iter = [0]*2
    for i in range(labels.shape[0]):
       label = labels[i]
       data[label][l_iter[label]] = images[i] / 255.
       l_iter[label] += 1
    #data[label][l_iter]=images/255
    return data

  def sample_pair(self, batch_size, label=None):
    label = 0#np.random.randint(10) if label is None else label
    images = self.data[label]
    
    #choice1 = np.random.choice(images.shape[0], batch_size)
    #choice2 = np.random.choice(images.shape[0], batch_size)
    choice1 = np.zeros([1],dtype=np.int)
    choice2 = np.ones([1], dtype=np.int)

    x = images[choice1]
    y = images[choice2]

    return x, y
