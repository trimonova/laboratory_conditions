import numpy as np
import matplotlib.pyplot as plt
#from ..input_parameters import delta_fi_list, delta_r_list, r_well, N_r_full, M_fi_full
#from ..find_pore_pressure_during_injection_QinPackers import viscosity_matrix
from input_parameters import delta_fi_list, delta_r_list, r_well, N_r_full, M_fi_full, X_matrix, Y_matrix, q
Pres_distrib_in_Time = np.load('Pres_distrib_in_Time_2.npy')
Func_matrix_remake_in_Time = np.load('Func_matrix_remake_in_Time.npy')
Func_matrix_in_Time = np.load('Func_matrix_in_Time.npy')
viscosity_in_Time = np.load('viscosity_in_Time.npy')
velocity_in_Time = np.load('velocity_in_Time.npy')
bound_coords_cart_in_Time = np.load('bound_coords_cart_in_Time.npy', allow_pickle=True)
print(bound_coords_cart_in_Time[0])
#print(bound_coords_cart_in_Time[10])
#print(bound_coords_cart_in_Time[20])
#print(bound_coords_cart_in_Time[30])
#print(bound_coords_cart_in_Time[50])
#print(bound_coords_cart_in_Time[59])
print(bound_coords_cart_in_Time[-1])

surf = plt.contourf(X_matrix, Y_matrix, Pres_distrib_in_Time[1], cmap=plt.get_cmap('jet'))
plt.title('pressure_first')
plt.colorbar(surf)
plt.show()

surf = plt.contourf(X_matrix, Y_matrix, Pres_distrib_in_Time[-1], cmap=plt.get_cmap('jet'))
plt.title('pressure_last')
plt.colorbar(surf)
plt.show()

surf = plt.contourf(X_matrix, Y_matrix, viscosity_in_Time[1], cmap=plt.get_cmap('jet'))
plt.title('visc_first')
plt.colorbar(surf)
plt.show()

surf = plt.contourf(X_matrix, Y_matrix, viscosity_in_Time[-1], cmap=plt.get_cmap('jet'))
plt.title('visc_last')
plt.colorbar(surf)
plt.show()

surf = plt.contour(X_matrix, Y_matrix, Func_matrix_remake_in_Time[0], levels = 0)
plt.title('Func_matrix_0')
#plt.colorbar(surf)
#plt.show()

#surf = plt.contour(X_matrix, Y_matrix, Func_matrix_remake_in_Time[5], levels = 0, cmap=plt.get_cmap('jet'))
#plt.title('Func_matrix_5')
#plt.colorbar(surf)
#plt.show()

#plt.contour(X_matrix, Y_matrix, Func_matrix_remake_in_Time[10], levels = 0)
#plt.title('Func_matrix_10')
#plt.colorbar(surf)
#plt.show()

plt.contour(X_matrix, Y_matrix, Func_matrix_remake_in_Time[2], levels=0)
#plt.title('Func_matrix_19')
#plt.colorbar(surf)
#plt.show()

plt.contour(X_matrix, Y_matrix, Func_matrix_remake_in_Time[-1], levels=0)
#plt.title('Func_matrix_last')
#plt.colorbar(surf)
plt.show()

surf = plt.contourf(X_matrix, Y_matrix, velocity_in_Time[1], cmap=plt.get_cmap('jet'))
plt.title('velocity_1')
plt.colorbar(surf)
plt.show()

surf = plt.contourf(X_matrix, Y_matrix, velocity_in_Time[-1], cmap=plt.get_cmap('jet'))
plt.title('velocity_last')
plt.colorbar(surf)
plt.show()

for coord_pair in bound_coords_cart_in_Time[1]:
    plt.scatter(coord_pair[0], coord_pair[1])
plt.show()

for coord_pair in bound_coords_cart_in_Time[-1]:
    plt.scatter(coord_pair[0], coord_pair[1])
plt.show()

print(velocity_in_Time[-1][:,10])
print(velocity_in_Time[1][:,10])
plt.plot(velocity_in_Time[-1][:,10])
plt.plot(velocity_in_Time[0][:,10])
plt.plot(velocity_in_Time[1][:,10])
plt.plot(velocity_in_Time[2][:,10])
plt.plot(velocity_in_Time[3][:,10])
# plt.plot(velocity_in_Time[10][:,10])
# plt.plot(velocity_in_Time[20][:,10])
# plt.plot(velocity_in_Time[40][:,10])
plt.scatter(0,-q)
plt.show()

#pressure_grad = np.gradient(Pres_distrib_in_Time, delta_r_list[0], delta_fi_list[0])
plt.title('pressure_grad')
plt.plot(np.gradient(Pres_distrib_in_Time[0], delta_r_list[0], delta_fi_list[0])[0][:,10])
#plt.plot(np.gradient(Pres_distrib_in_Time[60], delta_r_list[0], delta_fi_list[0])[0][:,10])
plt.plot(np.gradient(Pres_distrib_in_Time[-1], delta_r_list[0], delta_fi_list[0])[0][:,10])
plt.title('pressure_grad')
plt.show()

plt.plot(Func_matrix_remake_in_Time[0][:,10])
plt.plot(Func_matrix_in_Time[0][:,10])
plt.show()

plt.plot(Func_matrix_remake_in_Time[-1][:,10])
plt.plot(Func_matrix_in_Time[-1][:,10])
plt.show()

