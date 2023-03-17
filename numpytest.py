import numpy as np

for i in range(0,4):
    print('i',i)

a = np.arange(15).reshape(15,1)
b = [1,2,3,3,4,4,5,5]
print('a array shape',a.shape)
print('a array shape',a)

print('exp ',np.exp(2))
# windows = np.transpose(np.array(3), (0, 2, 1))
print('b',np.array(b))


c = np.transpose(b, (0, 2, 1))
print('a trans shape',c.shape)
print('a trans shape',c)

# a = np.arange(15).reshape(3, 5)
# print('a reshape array',a)
