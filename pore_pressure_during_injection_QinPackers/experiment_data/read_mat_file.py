import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.io
#f = scipy.io.loadmat('data2.mat')
f = h5py.File('data2.mat', 'r')

pressure_dict = {}
keys_list = list(f.keys())

for elem in keys_list:
    k_value = np.array(f.get(elem))
    pressure_dict[elem] = k_value
    print(elem)
print(pressure_dict['xp'])
print(pressure_dict['yp'])
print(pressure_dict['yhole'])
print(pressure_dict['xhole'])
print(np.shape(pressure_dict['p']))
pressure = pressure_dict['p'].transpose()*10**6



points_coords_dict = {0:[57, -127], 1:[70, 0], 2:[-57,127], 3: [0, 127], 4: [121,	-121], 5: [65, 65], 6: [0, 70],
7: [0, -185], 8: [127,	0], 9: [0, -70], 10: [0, 0], 11:[-57, -127], 12:[57, 127], 13:[-121, 121]}
points_coords_dict_rad = {}
for elem in points_coords_dict:
    x = points_coords_dict[elem][0]/1000
    y = points_coords_dict[elem][1]/1000
    if np.arctan2(y, x) >= 0:
        points_coords_dict_rad[elem] = [(x**2+y**2)**0.5, np.arctan2(y,x)]
    else:
        points_coords_dict_rad[elem] = [(x ** 2 + y ** 2) ** 0.5, 2*np.pi + np.arctan2(y, x)]

pressure = pressure_dict['p'].transpose()
P_filter_0 = savgol_filter(pressure[:,0], polyorder=3, window_length=151)
#plt.plot(pressure[:, 0])
#plt.plot(P_filter_0)
#plt.axis([0, 700000, 0, 4*10**5])
#plt.show()

P_filter_1 = savgol_filter(pressure[:,1], polyorder=3, window_length=5051)
# plt.plot(pressure[:, 1])
# plt.plot(P_filter_1)
# #plt.axis([0, 700000, 0, 4*10**5])
# plt.show()

P_filter_2 = savgol_filter(pressure[:,2], polyorder=3, window_length=151)
# plt.plot(pressure[:, 11])
# #plt.plot(P_filter_11)
# plt.axis([0, 700000, 0, 4*10**5])
# plt.show()

P_filter_3 = savgol_filter(pressure[:,3], polyorder=3, window_length=151)
# plt.plot(pressure[:, 21])
# #plt.plot(P_filter_21)
# plt.axis([0, 700000, 0, 4*10**5])
# plt.show()

#print(pressure_dict['flow'])
#print(points_coords_dict_rad)
P_filter_4 = savgol_filter(pressure[:,4], polyorder=3, window_length=151)
P_filter_5 = savgol_filter(pressure[:,5], polyorder=3, window_length=151)
P_filter_6 = savgol_filter(pressure[:,6], polyorder=3, window_length=151)
P_filter_7 = savgol_filter(pressure[:,7], polyorder=3, window_length=151)
P_filter_8 = savgol_filter(pressure[:,8], polyorder=3, window_length=151)
P_filter_9 = savgol_filter(pressure[:,9], polyorder=3, window_length=151)
P_filter_10 = savgol_filter(pressure[:,10], polyorder=3, window_length=151)
P_filter_11 = savgol_filter(pressure[:,11], polyorder=3, window_length=151)
P_filter_12 = savgol_filter(pressure[:,12], polyorder=3, window_length=151)
P_filter_13 = savgol_filter(pressure[:,13], polyorder=3, window_length=151)

plt.plot(pressure[:,10])
plt.plot(P_filter_10)
plt.show()

