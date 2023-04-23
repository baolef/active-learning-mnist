import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = np.load('svm.npy')
# print(data)

C_poly = data[:, 1, :]
C_kernel = data[:, :, 0]

for i in range(C_kernel.shape[0]):
    C_kernel[i, 1] = max(C_poly[i])

print(C_poly)
print(C_kernel)

C = [0.1,1,10,100,1000,10000]
Kernel = ['linear', 'poly', 'rbf', 'sigmoid']
Degree = [1, 3, 5, 10]


# sns.heatmap(C_kernel, annot=True, linewidths=0.6, xticklabels=Kernel, yticklabels=C, fmt='.3g')
# plt.title('Parameter Search for C and kernel')
# plt.xlabel('Kernel')
# plt.ylabel('C')
# plt.savefig('kernel.png')

sns.heatmap(C_poly, annot=True, linewidths=0.6, xticklabels=Degree, yticklabels=C, fmt='.3g')
plt.title('Parameter Search for poly')
plt.xlabel('Degree')
plt.ylabel('C')
plt.savefig('poly.png')
plt.show()

plt.show()
