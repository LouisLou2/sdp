import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd

import preprocess.prep as prep
from model.model import FeatureExtractor, Classifier, SupConLoss

dataset_filename = '../dataset/PC5.parquet'
data = pd.read_parquet(dataset_filename)
X_train, X_test, X_validation, y_train, y_test, y_validation = prep.split_data_with_oversampling(data)

# 将 DataFrame 转换为张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_train_tensor = y_train_tensor.squeeze()

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
y_test_tensor = y_test_tensor.squeeze()

X_validation_tensor = torch.tensor(X_validation.values, dtype=torch.float32)
y_validation_tensor = torch.tensor(y_validation.values, dtype=torch.long)
y_validation_tensor = y_validation_tensor.squeeze()

# 创建训练、测试和验证的 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


input_dim = X_train.shape[1]  # 根据你的数据设置输入维度
feature_dim = 30  # 特征维度，可以根据需要调整
num_classes = 2  # 二分类任务

# 创建特征提取器和分类器
feature_extractor = FeatureExtractor(input_dim, feature_dim)
classifier = Classifier(feature_dim, num_classes)

# 定义损失函数
criterion_contrastive = SupConLoss()
criterion_classifier = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=1e-3)


# 训练函数
def train_model(train_loader, feature_extractor, classifier, criterion_contrastive, criterion_classifier, optimizer,
                num_epochs=10):
    feature_extractor.train()
    classifier.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count=0
        for batch_x, batch_y in train_loader:
            # 前向传播：特征提取
            features = feature_extractor(batch_x)

            # 对比学习损失
            loss_contrastive = criterion_contrastive(features, batch_y)

            # 分类器前向传播
            logits = classifier(features)
            loss_classifier = criterion_classifier(logits, batch_y)

            # 总损失
            loss = loss_contrastive + loss_classifier

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # print(f'Epoch: {epoch} Batch:{batch_count} loss_contrastive: {loss_contrastive} loss_classifier: {loss_classifier}')
            batch_count+=1
        print(f"@@@@@@@@Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# 调用训练
train_model(train_loader, feature_extractor, classifier, criterion_contrastive, criterion_classifier, optimizer)

# 模型评估函数
def evaluate_model(data_loader, feature_extractor, classifier):
    feature_extractor.eval()
    classifier.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            features = feature_extractor(batch_x)
            logits = classifier(features)
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            tmp = (predicted == batch_y)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total * 100
    return accuracy

# 在测试集和验证集上评估模型
test_accuracy = evaluate_model(test_loader, feature_extractor, classifier)
validation_accuracy = evaluate_model(validation_loader, feature_extractor, classifier)

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Validation Accuracy: {validation_accuracy:.2f}%")