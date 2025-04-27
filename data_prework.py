# reference page: https://blog.csdn.net/qq_36018871/article/details/121361027
import argparse
from os.path  import join
import MnistDataloader as MD
import random
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")

parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')

# 客户端的数量
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')

# 隨機挑選的客户端的數量
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')

# 訓練次數(客户端更新次數)
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')

# batchsize大小
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')

# 模型名稱
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')

# 學習率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-dataset',"--dataset",type=str,default="mnist",help="需要訓練的資料集")

# 模型驗證頻率（通信頻率）
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')

#n um_comm 表示通信次數，此處設置為1k
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')

#載入資料集
def GetDataSet(train_data_size, test_data_size, client_amount, is_IID=True):
  data_dir = './mnist'
  # python路徑拼接os.path.join() 路徑變為.\data\MNIST\train-images-idx3-ubyte.gz
  train_images_path = join(data_dir, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
  train_labels_path = join(data_dir, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
  test_images_path = join(data_dir, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
  test_labels_path = join(data_dir, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

  mnist_dataloader = MD.MnistDataloader(train_images_path,train_labels_path,test_images_path,test_labels_path)
  (train_images, train_labels), (test_images, test_labels) = mnist_dataloader.load_data()
  '''
  # -------- 測試隨機印出一張train data的圖，如想測試需import matplotlib.pyplot as plt ---------
  r = random.randint(1, 60000)    
  plt.title('training image ['+ str(r) +']', fontsize=10)
  plt.imshow(train_images[r], cmap=plt.cm.gray)     #show random training image
  plt.show()
  '''
  #將圖片展平成一維向量，shape[0]是dataset圖片張數，shape[1]跟shape[2]是本來照片的size(28*28)，這裡把它攤平成784維(28*28=784)
  train_images = np.array(train_images)
  test_images = np.array(test_images)
  train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])   
  test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])


  #將像素數值都改成同一格式(np.float)
  #把像素值歸一化到0-1間(本來應該是0-255)
  #通過歸一化，可以使數值範圍更小且更集中，進一步加快收斂速度
  train_images = train_images.astype(np.float32)
  train_images = np.multiply(train_images, 1.0 / 255.0)   # 數組隊硬元素位置相乘
  test_images = test_images.astype(np.float32)
  test_images = np.multiply(test_images, 1.0 / 255.0)

  client_amount=10
  num_classes=10
  samples_per_client = train_data_size // client_amount  # 每個客戶端的樣本數
  # 初始化 train_data 和 train_label 為 NumPy 陣列
  client_train_data = np.zeros((client_amount, train_data_size // client_amount, 28*28), dtype=np.float32)
  client_train_label = np.zeros((client_amount, train_data_size // client_amount), dtype=np.int64)
  train_labels = np.array(train_labels, dtype=np.int64) # 下一行的np.where只能在np.array上執行 所以要先轉過來
  if not is_IID:
    # 類別索引，把相同類別的 index 放一起，這是一個字典，索引值是i(0~9)，每一個索引值會回傳的是在train_labels=i的index(也就是[0])，如此就收集到每一種數字分別的index
    class_indices = {i: np.where(train_labels == i)[0] for i in range(num_classes)}
    # selected_labels是 client_amount 個Numpy陣列組成的list，依照標籤來切，代表每個客戶端需要負責哪些標籤的資料(例如指有三個client的話，第一個client就要負責0 1 2三個標籤的資料)
    selected_labels = np.array_split(np.arange(num_classes), client_amount)
  
    # 將對應的類別數據分配給每個客戶端
    for i, labels in enumerate(selected_labels):
      # 為每個客戶端分配數據
      current_indices = []
      for label in labels:
          indices = class_indices[label]
          np.random.shuffle(indices)  # 隨機打亂索引
          # 計算需要從該標籤中選取的樣本數
          samples_needed = samples_per_client // len(labels)
          current_indices.extend(indices[:samples_needed])
      # 確保分配的樣本數不超過預分配空間
      current_indices = current_indices[:samples_per_client]
      print("Data amount of number ",i," is:",len(current_indices))
      # 如果樣本不足，隨機補充其他標籤的數據
      if len(current_indices) < samples_per_client:
          remaining = samples_per_client - len(current_indices)
          other_indices = np.setdiff1d(np.arange(train_data_size), current_indices)
          np.random.shuffle(other_indices)
          current_indices.extend(other_indices[:remaining])
      
      # 將數據分配給客戶端
      client_train_data[i] = train_images[current_indices]
      client_train_label[i] = train_labels[current_indices]
  else:
    # IID 分割
    indices = np.random.permutation(train_data_size)
    samples_per_client = train_data_size // client_amount
    for i in range(client_amount):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_train_data[i] = train_images[indices[start_idx:end_idx]]
        client_train_label[i] = train_labels[indices[start_idx:end_idx]]

  return (client_train_data, client_train_label), (test_images,test_labels)