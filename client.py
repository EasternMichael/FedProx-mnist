import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import data_prework
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Client:
    def __init__(self, dataset='mnist', learning_rate=0.01, batch_size=32, local_epoch=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.set_model().to(self.device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.local_epoch = local_epoch
        self.global_model_params = None
        self.mu = 0.1  #越大本地更新力度越小 aka 不會離全域太遠

    def set_model(self):
        # # ------ simple model -------
        # return nn.Sequential(
        #     nn.Linear(784, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 10)
        #     # Softmax layer ignored since the loss function defined is nn.CrossEntropy()
        # )

        # -------- strong model --------
        model = nn.Sequential(
            nn.Linear(784, 256),  # 增加第一隱藏層的神經元數
            nn.BatchNorm1d(256),  # 加入批次正規化
            nn.ReLU(),
            nn.Dropout(0.2),      # 加入Dropout，比例0.2
            nn.Linear(256, 128),  # 第二隱藏層
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),   # 第三隱藏層
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)     # 輸出層
            # Softmax層省略，因為損失函數是nn.CrossEntropyLoss()
        )
        
        # 使用He初始化layer m權重：假設輸入數據的方差為 1，He 初始化計算權重的方差，使每一層輸出的方差接近輸入的方差
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
        return model

    def train(self):
        """
        執行訓練並僅打印最後的 loss 和訓練準確率，訓練中加入 FedProx 的近端正則項（Proximal Term，不讓本地模型離全局模型太遠）。
        """
        self.model.train()  # 將模型設為訓練模式
        total_loss = 0      # 紀錄總損失
        correct = 0         # 紀錄正確分類的樣本數
        total_samples = 0   # 紀錄總樣本數

        # 開始批次訓練
        for batch, (X, y) in enumerate(self.data_loader):
            X, y = X.to(self.device), y.to(self.device)
            self.model.to(self.device)

            # 向前傳播
            y_pred = self.model(X)          #這裡的y_pred其實是logits，也就是預測結果的信心數值，沒有經過softmax layer的歸一化處理所以不一定會在[0,1]中
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()  # 累加批次損失

            # 反向傳播並更新參數
            self.optimizer.zero_grad()      # 將梯度清0 (因為backward()功能是將梯度蕾接到.grad，所以要清零)

            # FedProx 正則項
            if self.global_model_params is not None:
                prox_term = 0.0
                for param, global_param in zip(self.model.parameters(), self.global_model_params.values()):
                    prox_term += torch.norm(param - global_param, p=2)**2       #這一項是模型參數跟全局模型參數的距離
                loss += (self.mu / 2) * prox_term

            loss.backward()                         # 反向傳播計算每個參數的梯度，並會自動微分(為了提出梯度的"方向")後儲存到model.parameters的.grad屬性
            self.optimizer.step()                   # 根據梯度更新模型參數

            # 計算正確分類數
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            total_samples += y.size(0)

        # 訓練完成後計算平均損失和準確率
        avg_loss = total_loss / len(self.data_loader)
        train_accuracy = correct / total_samples * 100

        return avg_loss, train_accuracy

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_params):
        self.model.load_state_dict(model_params)

    def set_global_model_params(self, global_model_params):
        self.model.load_state_dict(global_model_params)

'''
# 單個client執行

if __name__ == '__main__':
    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)

    # 載入測試資料
    test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

    local_data = train_data[0]
    local_label = train_label[0]
    local_data_tensor = torch.tensor(local_data)
    local_label_tensor = torch.tensor(local_label)
    client1=Client(dataset=TensorDataset(local_data_tensor, local_label_tensor))
    print(client1.model)

    for i in range(30):
        client1.train()
        print("test acc of epoch ",i,": ",client1.evaluate(test_data_loader))
    
    print("final test acc: ",client1.evaluate(test_data_loader))
'''
