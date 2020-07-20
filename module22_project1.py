import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)
fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()


#Recogninising hand written code


import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)
new_samples = np.array([
[0.00,6.71,7.09,6.10,6.10,6.10,4.96,0.00,0.00,6.10,2.44,2.44,3.05,3.05,2.59,0.15,0.00,5.03,7.47,5.87,4.58,4.58,6.33,4.04,0.00,0.69,1.60,0.00,0.00,0.00,1.22,6.86,0.00,0.00,0.00,0.00,0.00,0.00,0.76,6.86,1.15,3.28,0.00,0.00,0.00,0.23,5.04,5.42,1.68,7.02,4.58,4.65,5.34,7.09,5.65,0.46,0.08,2.82,3.05,3.05,2.44,1.37,0.00,0.00],
[0.00,3.81,4.27,2.44,1.53,2.14,1.83,0.00,0.00,1.45,3.89,5.87,6.10,6.71,4.58,0.00,0.00,0.00,0.00,0.00,0.00,3.05,4.58,0.00,0.00,0.00,0.00,0.00,0.00,4.27,4.19,0.00,0.00,0.00,0.00,0.00,0.08,6.48,1.83,0.00,0.00,0.00,0.00,0.00,1.83,6.56,0.08,0.00,0.00,0.00,0.00,0.00,3.81,4.50,0.00,0.00,0.00,0.00,0.00,0.00,6.79,2.44,0.00,0.00],
[0.00,0.00,1.37,6.86,7.47,7.55,2.90,0.00,0.00,0.00,0.00,0.76,0.30,2.21,6.56,0.00,0.00,0.00,0.00,0.00,0.00,0.00,7.09,0.76,0.00,0.00,0.00,0.00,0.00,3.51,6.87,0.31,0.00,0.00,0.00,0.00,2.22,6.71,1.22,0.00,0.00,0.00,0.00,1.22,6.86,6.26,4.58,4.50,0.00,0.00,0.00,2.75,4.73,3.20,3.05,2.97,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,1.76,7.47,2.60,0.00,0.00,0.00,0.23,4.73,7.17,6.03,3.81,0.00,0.00,0.00,3.51,6.18,0.46,2.98,5.19,0.00,0.00,0.53,6.86,1.98,0.00,1.60,6.56,0.00,0.00,5.95,7.55,5.34,6.63,6.94,6.86,0.00,0.00,3.36,3.05,2.29,1.75,0.92,7.48,0.38,0.00,0.00,0.00,0.00,0.00,0.00,6.86,1.15,0.00,0.00,0.00,0.00,0.00,0.00,5.57,2.44,0.00]
])

new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')



