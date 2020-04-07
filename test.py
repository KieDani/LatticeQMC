import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

mpl.rc('axes', labelsize='x-large', titlesize='xx-large', )
mpl.rc('xtick', labelsize='x-large')
mpl.rc('ytick', labelsize='x-large')
mpl.rc('legend', fontsize='x-large')
mpl.rc('figure', titlesize='xx-large')

# mu = None
# mu = 1 if mu is not None else 2
# print(mu)
# something = f"blalba"
# print(type(something))

# v = np.zeros((3,3))
# a = np.array(range(1,4))
# print(v)
# print(a)
# np.fill_diagonal(v, a)
# print(v)
# print(a[-1])
# for i in range(0, 0):
#     print('ollah', i)

filenames = ["U=0.5", "U=1", "U=2", "U=4", "U=6", "U=8", "U=10", "U=12"]
filepath = "E:\\Download\\5.Semester\\Computational_Physics\\project"

path = os.path.join(filepath, filenames[0])
data = pd.read_csv(path, sep=",", header=None)
temps = data.iloc[:,0].copy().values[:]
datatable = pd.DataFrame(columns=filenames)
datatable['Temperatures'] = temps

for filename in filenames:
    path = os.path.join(filepath, filename)
    data = pd.read_csv(path, sep=",", header=None)
    datatable[filename] = data.iloc[:,1].copy().values[:]

print(datatable)
datatable.plot(x='Temperatures', y=filenames, xlim=(1,20.5), ylim=(0.45,1.05))
plt.xlabel('Temperature')
plt.ylabel('Filling')
plt.legend()
plt.show()
