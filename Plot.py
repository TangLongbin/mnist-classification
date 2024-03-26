# The code is programmed by 
# Tang Longbin and Shi Jianrui
# for EES4408 Final Project.
import matplotlib.pyplot as plt

# Logistic Regression Performance
learning_rates = [1e-4, 1e-3, 1e-2]
epochs_lr = [3, 5, 10, 20, 50, 100]
accuracy_lr = [
    [0.8938, 0.9064, 0.9146, 0.9217, 0.9257, 0.9288],
    [0.9234, 0.9251, 0.9261, 0.9281, 0.9284, 0.9279],
    [0.9166, 0.9110, 0.8959, 0.9188, 0.9143, 0.9111]
]

# SVM Performance
epochs_svm = [3, 5, 10, 15, 20, 50, 100, 200, 500, 1000, 2000, 5000]
accuracy_svm = [
    [0.4475, 0.5778, 0.5162, 0.7126, 0.7525, 0.8818, 0.9271, 0.9697, 0.9797, 0.979, 0.9795, 0.9792]
]

# kNN Performance
k_values = [3, 5, 7, 9, 15, 27, 49]
accuracy_knn = [0.9720, 0.9720, 0.9708, 0.9705, 0.9664, 0.9611, 0.9526]

# Plotting Logistic Regression
plt.figure(figsize=(10, 6))

for i, lr in enumerate(learning_rates):
    plt.plot(epochs_lr, accuracy_lr[i], label=f"LR (Learning Rate={lr})")

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Performance')
plt.legend()
plt.grid(True)
plt.show()

# Plotting SVM with log scale
plt.figure(figsize=(10, 6))

plt.semilogx(epochs_svm, accuracy_svm[0], label="SVM")

plt.xlabel('Log(Epochs)')
plt.ylabel('Accuracy')
plt.title('SVM Performance')
plt.legend()
plt.grid(True)
plt.show()

# Plotting kNN
plt.figure(figsize=(10, 6))

plt.plot(k_values, accuracy_knn, label="kNN")

plt.xlabel('k Values')
plt.ylabel('Accuracy')
plt.title('kNN Performance')
plt.legend()
plt.grid(True)
plt.show()
