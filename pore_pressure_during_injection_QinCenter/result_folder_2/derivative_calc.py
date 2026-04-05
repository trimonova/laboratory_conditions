import numpy as np
import matplotlib.pyplot as plt

from input_parameters import delta_r_list, delta_fi_list, perm
print(delta_r_list[0])

Pres_distrib_in_Time = np.load('Pres_distrib_in_Time_2.npy')
Func_matrix_remake_in_Time = np.load('Func_matrix_remake_in_Time.npy')
#Func_matrix_in_Time = np.load('Func_matrix_in_Time.npy')
viscosity_in_Time = np.load('viscosity_in_Time.npy')
velocity_in_Time = np.load('velocity_in_Time.npy')

time_step = -1
angle = 5

# давление вдоль одного угла
pressure_list = Pres_distrib_in_Time[time_step][:, angle]
plt.plot(pressure_list)
plt.show()

func_matrix_list = Func_matrix_remake_in_Time[time_step][:, angle]

# градиент давления через numpy, как у меня в программе
pressure_der_numpy = np.gradient(Pres_distrib_in_Time[time_step], delta_r_list[0], delta_fi_list[0])

# нахожу градиент сама:

pressure_der_center = [(pressure_list[i+1]-pressure_list[i-1])/2/delta_r_list[0] for i in range(1, len(pressure_list)-1)]
pressure_der = [(pressure_list[1]-pressure_list[0])/delta_r_list[0]]
pressure_der = pressure_der + pressure_der_center
pressure_der.append((pressure_list[-1]-pressure_list[-2])/delta_r_list[0])

# сравниваем градиенты. Одинаковые получились.
plt.plot(pressure_der)
plt.plot(pressure_der_numpy[0][:, angle])
plt.show()

viscosity_list = viscosity_in_Time[time_step][:, angle]
plt.plot(viscosity_list)
plt.show()

pressure_der_center_new = []
for i in range(1, len(pressure_list)-1):
    if func_matrix_list[i] > 0 and func_matrix_list[i+1] < 0:
        pressure_der_center_new.append((pressure_list[i]-pressure_list[i-1])/delta_r_list[0])
    elif func_matrix_list[i] < 0 and func_matrix_list[i-1] > 0:
        pressure_der_center_new.append((pressure_list[i+1] - pressure_list[i]) / delta_r_list[0])
    else:
        pressure_der_center_new.append((pressure_list[i + 1] - pressure_list[i-1]) / 2/delta_r_list[0])

pressure_der_new = [(pressure_list[1] - pressure_list[0]) / delta_r_list[0]]
pressure_der_new = pressure_der_new + pressure_der_center
pressure_der_new.append((pressure_list[-1] - pressure_list[-2]) / delta_r_list[0])

plt.plot(pressure_der_new)
plt.plot(pressure_der_numpy[0][:, angle])
plt.show()