#import matplotlib.pyplot as plt
# from train import train_accuracy, test_accuracy,train_Loss,test_Loss


import matplotlib.pyplot as plt
list = [1,2,3,4,5]
train_accuracy = [14,9,8,6,4]
test_accuracy = [13,11,7,5,3]
plt.plot(list,train_accuracy, c = "g", label = "Train Accuracy")
plt.plot(list,test_accuracy,c ='r', label ="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
