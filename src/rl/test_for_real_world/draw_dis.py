import numpy as np
import matplotlib.pyplot as plt

x = np.load('./distributed_x_.npy')
y = np.load('./distributed_y_.npy')
tanhx = np.load('./distributed_tanhx_.npy')

for _ in range(len(x)//2):
    # plt.plot(x[_], y[_], label='Gaussian distribution')
    plt.axvline(x=tanhx[-_], color='r', linestyle='--', label='Mean')
    print(tanhx[_])
# plt.xlim(-100, 100)
plt.xlim(-3, 3)
plt.show()