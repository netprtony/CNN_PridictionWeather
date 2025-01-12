import string
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import joblib


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load mô hình đã lưu
model_cnn = load_model('model-v1.h5')
model_vgg19 =load_model("vgg19.keras")
model_svm = joblib.load('weather_prediction_svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Khởi tạo ứng dụng Tkinter
root = tk.Tk()
root.title("Image Classifier")

# Khung để hiển thị hình ảnh
panel = tk.Label(root)
panel.pack(padx=50, pady=50)
root.geometry("800x600")
# Khung để hiển thị kết quả dự đoán
result_labelCNN = tk.Label(root, text="CNN Model Prediction: ")
result_labelCNN.pack(pady=10)
result_labelVGG19 = tk.Label(root, text="VGG19 Model Prediction: ")
result_labelVGG19.pack(pady=10)
result_labelSVM = tk.Label(root, text="SVM model Prediction: ")
result_labelSVM.pack(pady=10)


def names(number):
    if number==0:
        return "cloudy"
    elif number==1:
        return "foggy"
    elif number==2:
        return "rainy"
    elif number==3:
        return "shine"
    elif number==4:
        return "sunrise"
def Prediction(im, model):
    x = img_to_array(im) / 255.0
    x = np.expand_dims(x, axis = 0)
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    res =  str(res[0][classification]*100) + '% Confidence ' + names(classification)
    return res

    
def predict_weather_svm(image_path, model, label_encoder):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    resized_image = cv2.resize(image, (150, 150))
    image_array = resized_image.astype('float32') / 255.0  # Normalize pixel values
    image_array = image_array.reshape(1, -1)  # Flatten image

    prediction = model.predict(image_array)

    predicted_class = label_encoder.inverse_transform(prediction)[0]
    return predicted_class

def load_image():
    # Mở hộp thoại để chọn hình ảnh
    file_path = filedialog.askopenfilename()
    if file_path:
    # Mở và hiển thị hình ảnh
        img = Image.open(file_path).resize((150, 150))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        # Tiền xử lý hình ảnh
        img = Image.open(file_path).resize((150, 150))
        result_labelCNN.config(text=f"CNN Model Prediction:{Prediction(img, model_cnn)}")

    if file_path:
    # Mở và hiển thị hình ảnh
        img = Image.open(file_path).resize((224, 224))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        # Tiền xử lý hình ảnh
        img = Image.open(file_path).resize((224, 224))
        result_labelVGG19.config(text=f"VGG19 Model Prediction:{Prediction(img, model_vgg19)}")

    if file_path: 
        img = Image.open(file_path).resize((224, 224))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        # Tiền xử lý hình ảnh
        img = Image.open(file_path).resize((150, 150))
        result_labelSVM.config(text=f"SVM model Prediction: {predict_weather_svm(file_path, model_svm, label_encoder)}")
    

# Nút để chọn hình ảnh
select_button = tk.Button(root, text="Select Image", command=load_image)
select_button.pack(pady=10)

# Chạy vòng lặp sự kiện Tkinter
root.mainloop()