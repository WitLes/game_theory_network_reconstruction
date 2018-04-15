import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
data = np.random.normal(size=(1000, 5), loc=0, scale=1)
print(data)
labels = [str(i) for i in range(5)]
plt.boxplot(data, labels=labels)

plt.show()
