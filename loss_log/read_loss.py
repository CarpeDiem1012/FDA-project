import scipy.io

data = scipy.io.loadmat('./loss_log_01.mat')
print('Number of data:')
print(data['loss_train'].reshape(-1).shape)
# print(data.keys())
# print(data['loss_train'].shape)
# print(type(data['loss_train']))

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(data['loss_train'].reshape(-1), label='train_loss')
plt.plot(data['loss_val'].reshape(-1), label='val_loss')
plt.legend()
plt.xlabel('steps//1000')
plt.ylabel('loss')
plt.title('model training')
fig.savefig('./loss_log_01.png')