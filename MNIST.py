import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train=np.reshape(x_train,(60000,784))/255.0

x_test=np.reshape(x_test,(10000,784))/255.0

y_train=np.matrix(np.eye(10)[y_train])
y_test=np.matrix(np.eye(10)[y_test])
print("---------------------------------")
print(x_train.shape)
print(y_train.shape)

#định nghĩa hàm ReLU
def relu(x):
  return np.maximum(x,0)
#Định nghĩa hàm sigmoid 
def sigmoid(x):
  return 1./(1.+np.exp(-x))
#Định nghĩa hàm softmax cho output layer
def softmax(x):
  return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))
def Forwardpass(X, Wh1, bh1, Wh2, bh2, Wo, bo):
  zh1 = X@Wh1.T +bh1
  ah1 = relu(zh1)
  zh2 = ah1@Wh2.T +bh2
  ah2= sigmoid(zh2)
  z = ah2@Wo.T +bo
  o = softmax(z)
  return o

def AccTest(label, prediction): #Tính toán độ chính xác của mô hình
  OutMaxArg = np.argmax(prediction, axis=1)
  LabelMaxArg = np.argmax(label, axis=1)  
  Accuracy = np.mean(OutMaxArg == LabelMaxArg) 
  return Accuracy

learningRate = 0.1 # Xác định learningRate
Epoch = 50 # Số lần lặp để tìm trọng số
NumTrainSamples = 60000 #Số mẫu trainig
NumTestSamples = 10000 #Số mẫu test

NumInputs = 784   #Số lượng ngõ vào
NumHiddenUnits = 512 
NumClasses = 10 #Số lượng class ngõ ra

#hidden layer 1
Wh1 = np.matrix(np.random.uniform(-0.5, 0.5, (NumHiddenUnits, NumInputs))) # tạo ma trận ngẫu nhiên
bh1 = np.random.uniform(0, 0.5, (1, NumHiddenUnits)) 
dWh1 = np.zeros((NumHiddenUnits, NumInputs)) #Tạo ma trận rỗng chứa trọng số
dbh1 = np.zeros((1, NumHiddenUnits)) 
#hidden layer 2
Wh2 = np.matrix(np.random.uniform(-0.5, 0.5, (NumHiddenUnits, NumHiddenUnits)))
bh2 = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWh2 = np.zeros((NumHiddenUnits, NumHiddenUnits))
dbh2 = np.zeros((1, NumHiddenUnits))
#Output layer
Wou = np.random.uniform(np.random.uniform(-0.5, 0.5, (NumClasses, NumHiddenUnits)))
bz = np.random.uniform(0, 0.5, (1, NumClasses))
dWou = np.zeros((NumClasses, NumHiddenUnits)) 
dbz = np.zeros((1, NumClasses))

from IPython.display import clear_output
loss = []
Acc = []
Batch_size = 200
Stochastic_samples= np.arange(NumTrainSamples)


for ep in range (Epoch):
    #feed forward proparagation

    np.random.shuffle(Stochastic_samples)
    
  
    for ite in range(0, NumTrainSamples, Batch_size):
      
        Batch_samples = Stochastic_samples[ite:ite + Batch_size]
    
        x=x_train[Batch_samples, :]
        y=y_train[Batch_samples, :]
        #Tính giá trị tại output layer 
        zh1 = x@Wh1.T + bh1
        ah1 = relu(zh1)
        zh2 = ah1@Wh2.T + bh2
        ah2 = sigmoid(zh2)
        z = ah2@Wou.T + bz
        o = softmax(z)

        #Cross-Entropy Loss  
        loss.append(-np.sum(np.multiply(y, np.log10(o))) / Batch_size)
      
        Eo = o-y

        #Back propagate error
        dh2 = Eo@Wou 
        Eh2 = np.multiply(np.multiply(dh2, ah2), (1-ah2))  
        Eh1 = np.where(zh1 > 0, np.matmul(Eh2, Wh2), 0)   
        
        #Cập nhật trọng số và bias tại ngõ ra
        dWou = np.matmul(np.transpose(Eo), ah2)
        dbz = np.mean(Eo)
        Wou = Wou - learningRate*dWou/Batch_size
        bz = bz - learningRate*dbz
        
        #Cập nhật trọng số và bias tại hidden layer 2
        dWh2 = np.matmul(np.transpose(Eh2), ah1)
        dbh2 = np.mean(Eh2)
        Wh2 = Wh2 - learningRate*dWh2/Batch_size
        bh2 = bh2 - learningRate*dbh2
        
        #Cập nhật trọng số và bias tại hidden layer 1
        dWh1 = np.matmul(np.transpose(Eh1), x)
        dbh1 = np.mean(Eh1)
        Wh1 = Wh1 - learningRate*dWh1/Batch_size
        bh1 = bh1 - learningRate*dbh1 

    #Kiểm tra độ chính xác của mô hình
    prediction = Forwardpass(x_test, Wh1, bh1, Wh2, bh2, Wou, bz)
    Acc.append(AccTest(y_test, prediction))
    clear_output(wait = True)
    plt.plot([i for i, _ in enumerate(Acc)], Acc, 'o')

    prediction = Forwardpass(x_test, Wh1, bh1, Wh2, bh2, Wou, bz)
    Rate = AccTest(y_test, prediction)
        
    print(ep,': ',Rate)
    plt.show(block=False)  
    plt.pause(0.1)  

prediction = Forwardpass(x_test, Wh1, bh1, Wh2, bh2, Wou, bz)
Rate = AccTest(y_test, prediction)
print("Result: ")
print(Rate)


ima = Image.open('anh.jpg')
plt.imshow(ima)
plt.axis('off')
plt.show()
gray_ima = ima.convert('L')
resized_ima = gray_ima.resize((28, 28))
plt.imshow(resized_ima, cmap='gray')
plt.axis('off')
plt.show()
normalized_ima = np.array(resized_ima) / 255.0
flattened_ima = normalized_ima.reshape(1, -1)
prediction = Forwardpass(flattened_ima, Wh1, bh1, Wh2, bh2, Wou, bz)
predicted_digit = np.argmax(prediction)

# Hiển thị kết quả dự đoán
print("Predicted digit:", predicted_digit)
