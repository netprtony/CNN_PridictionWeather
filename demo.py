import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from PIL import Image
import os

# Đường dẫn tới thư mục chứa các file hình ảnh
data_dir = 'D:\Work_space\MachineLearning\DemoWeather\dataset'

# Load và biểu diễn dữ liệu
X = []
y = []

for filename in os.listdir(data_dir):
    if filename.endswith('.png'):
        img = Image.open(os.path.join(data_dir, filename))
        img = img.resize((64, 64))  # Resize hình ảnh nếu cần
        img = np.array(img)
        X.append(img.flatten())  # Biến đổi hình ảnh thành vector đặc trưng
        label = filename.split('.')[0]
        y.append(label)

X = np.array(X)
y = np.array(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu: chuẩn hóa
scaler = StandardScaler()

# Mã hóa nhãn
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Tạo pipeline cho mô hình
model = make_pipeline(scaler, LogisticRegression(max_iter=1000))

# Huấn luyện mô hình
model.fit(X_train, y_train_encoded)

# Đánh giá mô hình trên tập kiểm tra
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, label_encoder.inverse_transform(y_pred))
print("Accuracy:", accuracy)