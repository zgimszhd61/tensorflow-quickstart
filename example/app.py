
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 创建一个人工数据集
n_samples = 1000
n_features = 10
n_actions = 5

# 状态数据，形状为 [n_samples, n_features]
X = np.random.rand(n_samples, n_features)

# 动作数据，形状为 [n_samples, n_actions]
y = np.random.randint(0, 2, size=(n_samples, n_actions))

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='linear')  # 输出层的激活函数取决于动作空间的类型
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 评估模型
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# 使用模型进行预测
some_state = np.random.rand(1, n_features)
pred = model.predict(some_state)
print(f"Predicted action: {pred}")

