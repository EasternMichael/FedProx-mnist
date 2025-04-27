import numpy as np  # 提供數組操作（如 np.random.permutation、np.vstack 等）
import torch  # 提供 tensor 操作（如 torch.tensor）
from torch.utils.data import DataLoader, TensorDataset  # 用於加載和封装數據集
import data_prework
import client

class ClientGroup:
    def __init__(self, num_of_clients=10, cfraction=0.6, epoch=30, batchsize=32, modelname='default_model', learning_rate=0.001, dataset_name='default_dataset', train_dataset_size=60000, test_dataset_size=10000, val_freq=1, save_freq=10, num_comm=5, save_path='./save', is_iid=True):
        self.num_of_clients = num_of_clients
        self.cfraction = cfraction      #每次訓練取的client數量
        self.epoch = epoch
        self.batchsize = batchsize
        self.modelname = modelname
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
        self.train_dataset_size = train_dataset_size
        self.test_dataset_size = test_dataset_size
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.num_comm = num_comm
        self.save_path = save_path
        self.is_iid = is_iid   
        self.train_data = [[] for _ in range(self.num_of_clients)]
        self.train_label = [[] for _ in range(self.num_of_clients)]
        self.test_data = []
        self.test_label = []
        self.test_data_loader = None
        self.clients_set = {}

myClients = ClientGroup(epoch=30, dataset_name='mnist', is_iid=True)

# 取得切好的資料集
(myClients.train_data, myClients.train_label),(myClients.test_data, myClients.test_label) = data_prework.GetDataSet(myClients.train_dataset_size, myClients.test_dataset_size, myClients.num_of_clients, myClients.is_iid)

test_data = torch.tensor(myClients.test_data)
test_label = torch.tensor(myClients.test_label)

# 載入測試資料
myClients.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
# TensorDataset 是 PyTorch 提供的一種資料集類別 (Dataset)，用於將 tensor 格式的資料與對應的標籤配對，形成一個可供模型訓練或評估的資料集。
# DataLoader 將 TensorDataset 封裝為一個可迭代物件，按指定的批量大小 (batch_size=100) 分批返回數據

# -------- 宣告初始global model --------
global_model=client.Client(dataset=TensorDataset(test_data, test_label))

# -------- 建立client set --------
# 逐個建立client
for i in range(myClients.num_of_clients):
    # 提取每個客戶端的數據和標籤
    local_data = myClients.train_data[i]
    local_label = myClients.train_label[i]

    # 將數據轉換為 PyTorch Tensor
    local_data_tensor = torch.tensor(local_data)
    local_label_tensor = torch.tensor(local_label)

    # 逐個建立client然後加入clients_set
    someone = client.Client(dataset=TensorDataset(local_data_tensor, local_label_tensor))
    someone.set_global_model_params(global_model.get_model_params())
    myClients.clients_set[i] = someone
# 到這邊把十個clients建立好後放到myClient裡面的clients_set裡面了，接下來應該要在下一段程式碼協同客戶端運作

global_model=client.Client(dataset=TensorDataset(test_data, test_label))

# -------- FL training --------
for round in range(myClients.epoch):
    # 每次隨機選取cfraction個Clients進行聚合
    num_in_comm = int(max(myClients.num_of_clients * myClients.cfraction, 1))

    # 得到被挑選的clients，亂序後取前num_in_comm個
    order = np.random.permutation(myClients.num_of_clients)
    clients_in_comm = []    # 放要聚合的clients編號
    for i in range(num_in_comm):
        clients_in_comm.append(order[i])

    clients_para=[]
    for i in clients_in_comm:       #local train挑出來的clients
        current_client=myClients.clients_set[i]
        current_client.train()
        clients_para.append(current_client.get_model_params())

    # 取平均值，得到本次通信中Server得到的更新后的模型参数
    global_para = clients_para[0]
    # state_dict 形式的字典(就是model_para啦)，直接相加和除法操作是無法直接進行的，需要逐層參數地進行操作
    for key in global_para.keys():              
        for i in range(1, num_in_comm):
            global_para[key] += clients_para[i][key]
        global_para[key] = global_para[key] / num_in_comm

    # 廣播聚合後model給clients
    for i in range(myClients.num_of_clients):
        myClients.clients_set[i].set_model_params(global_model.get_model_params())    
        myClients.clients_set[i].set_global_model_params(global_para)                   # 因為client訓練的時候要一直拿著當前global model算正則化項，所以要給一份全域模型
    
    # 更新global model以及算一次準確率   
    global_model.set_model_params(global_para)
    print("test accuracy of epoch ",round+1,":", global_model.evaluate(myClients.test_data_loader))


# 測試最終準確率
print("Final accuracy:", myClients.clients_set[0].evaluate(myClients.test_data_loader))
