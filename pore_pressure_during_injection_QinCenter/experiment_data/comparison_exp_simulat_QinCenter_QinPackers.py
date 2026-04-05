from read_mat_file import points_coords_dict_rad as points_r_fi_pair
from read_mat_file import P_filter_0, P_filter_1, P_filter_2, P_filter_3, P_filter_4, P_filter_5
from read_mat_file import P_filter_6, P_filter_7, P_filter_8, P_filter_9, P_filter_10, P_filter_11, P_filter_12, P_filter_13
from pore_pressure_during_injection_QinPackers.input_parameters import \
    delta_fi_list, delta_r_list, r_well, N_r_full, M_fi_full, X_matrix, Y_matrix, delta_r, delta_fi, delta_t, M_1
import numpy as np
import matplotlib.pyplot as plt
Pres_distrib_in_Time = np.load('../result_folder/Pres_distrib_in_Time_2_2.npy')
Func_matrix_remake_in_Time = np.load('../result_folder/Func_matrix_remake_in_Time_2.npy')
viscosity_in_Time = np.load('../result_folder/viscosity_in_Time_2.npy')
velocity_in_Time = np.load('../result_folder/velocity_in_Time_2.npy')
bound_coords_cart_in_Time = np.load('../result_folder/bound_coords_cart_in_Time_2.npy', allow_pickle=True)

points_coords = {}
for key in points_r_fi_pair:
    points_coords[key] = (int(round(points_r_fi_pair[key][0]/delta_r)), int(round(points_r_fi_pair[key][1]/delta_fi)))
#points_coords = [(round(i[0]/delta_r), round(i[1]/delta_fi)) for i in points_r_fi_pair]
print('points_coords', points_coords)

t_list = [j*delta_t for j in range(len(Pres_distrib_in_Time))]
t_list_exp = [i*0.01 for i in range(np.shape(P_filter_1)[0])]

PinPoint1 = Pres_distrib_in_Time[:, points_coords[1][0], points_coords[1][1]]
fig, axs = plt.subplots(3, 2, figsize=(21, 10))
axs[0, 0].set_title(str((round(points_r_fi_pair[1][0], 2), round(points_r_fi_pair[1][1], 2))), y = 0.75, loc='left')
axs[0, 0].set_xlabel("Time, s")
axs[0, 0].set_ylabel("Pressure, MPa")
axs[0, 0].plot(t_list_exp, P_filter_1)
axs[0, 0].plot(t_list, PinPoint1/10**6)

PinPoint2 = Pres_distrib_in_Time[:, points_coords[2][0], points_coords[2][1]]
axs[0, 1].set_title(str((round(points_r_fi_pair[2][0], 2), round(points_r_fi_pair[2][1], 2))), y = 0.75, loc='left')
axs[0, 1].set_xlabel("Time, s")
axs[0, 1].set_ylabel("Pressure, MPa")
axs[0, 1].plot(t_list_exp, P_filter_2)
axs[0, 1].plot(t_list, PinPoint2/10**6)

PinPoint10 = Pres_distrib_in_Time[:, points_coords[10][0], points_coords[10][1]]
PinCenter = Pres_distrib_in_Time[:, 1, M_1]
axs[2, 1].set_title(str((round(points_r_fi_pair[10][0], 2), round(points_r_fi_pair[10][1], 2))), y = 0.75, loc='left')
axs[2, 1].plot(t_list_exp, P_filter_10)
axs[2, 1].plot(t_list, PinPoint10/10**6)
axs[2, 1].plot(t_list, np.array(PinCenter)/10**6)

PinPoint6 = Pres_distrib_in_Time[:, points_coords[6][0], points_coords[6][1]]
axs[1, 1].set_title(str((round(points_r_fi_pair[6][0], 2), round(points_r_fi_pair[6][1], 2))), y = 0.75, loc='left')
axs[1, 1].set_xlabel("Time, s")
axs[1, 1].set_ylabel("Pressure, MPa")
axs[1, 1].plot(t_list_exp, P_filter_6)
axs[1, 1].plot(t_list, PinPoint6/10**6)

PinPoint7 = Pres_distrib_in_Time[:, points_coords[7][0], points_coords[7][1]]
axs[1, 0].set_title(str((round(points_r_fi_pair[7][0], 2), round(points_r_fi_pair[7][1], 2))), y = 0.75, loc='left')
axs[1, 0].set_xlabel("Time, s")
axs[1, 0].set_ylabel("Pressure, MPa")
axs[1, 0].plot(t_list_exp, P_filter_7)
axs[1, 0].plot(t_list, PinPoint7/10**6)

PinPoint9 = Pres_distrib_in_Time[:, points_coords[9][0], points_coords[9][1]]
axs[2, 0].set_title(str((round(points_r_fi_pair[9][0], 2), round(points_r_fi_pair[9][1], 2))), y = 0.75, loc='left')
axs[2, 0].set_xlabel("Time, s")
axs[2, 0].set_ylabel("Pressure, MPa")
axs[2, 0].plot(t_list_exp, P_filter_9)
axs[2, 0].plot(t_list, PinPoint9/10**6)

plt.show()

fig, axs = plt.subplots(3, 2, figsize=(21, 10))
PinPoint0 = Pres_distrib_in_Time[:, points_coords[0][0], points_coords[0][1]]
axs[0, 0].set_title(str((round(points_r_fi_pair[0][0], 2), round(points_r_fi_pair[0][1], 2))), y = 0.75, loc='left')
axs[0, 0].set_xlabel("Time, s")
axs[0, 0].set_ylabel("Pressure, MPa")
axs[0, 0].plot(t_list_exp, P_filter_0)
axs[0, 0].plot(t_list, PinPoint0/10**6)

PinPoint3 = Pres_distrib_in_Time[:, points_coords[3][0], points_coords[3][1]]
axs[0, 1].set_title(str((round(points_r_fi_pair[3][0], 2), round(points_r_fi_pair[3][1], 2))), y = 0.75, loc='left')
axs[0, 1].set_xlabel("Time, s")
axs[0, 1].set_ylabel("Pressure, MPa")
axs[0, 1].plot(t_list_exp, P_filter_3)
axs[0, 1].plot(t_list, PinPoint3/10**6)

PinPoint4 = Pres_distrib_in_Time[:, points_coords[4][0], points_coords[4][1]]
axs[2, 1].set_title(str((round(points_r_fi_pair[4][0], 2), round(points_r_fi_pair[4][1], 2))), y = 0.75, loc='left')
axs[2, 1].set_xlabel("Time, s")
axs[2, 1].set_ylabel("Pressure, MPa")
axs[2, 1].plot(t_list_exp, P_filter_4)
axs[2, 1].plot(t_list, PinPoint4/10**6)

PinPoint5 = Pres_distrib_in_Time[:, points_coords[5][0], points_coords[5][1]]
axs[1, 1].set_title(str((round(points_r_fi_pair[5][0], 2), round(points_r_fi_pair[6][1], 2))), y = 0.75, loc='left')
axs[1, 1].set_xlabel("Time, s")
axs[1, 1].set_ylabel("Pressure, MPa")
axs[1, 1].plot(t_list_exp, P_filter_5)
axs[1, 1].plot(t_list, PinPoint5/10**6)

PinPoint8 = Pres_distrib_in_Time[:, points_coords[8][0], points_coords[8][1]]
axs[1, 0].set_title(str((round(points_r_fi_pair[8][0], 2), round(points_r_fi_pair[7][1], 2))), y = 0.75, loc='left')
axs[1, 0].set_xlabel("Time, s")
axs[1, 0].set_ylabel("Pressure, MPa")
axs[1, 0].plot(t_list_exp, P_filter_8)
axs[1, 0].plot(t_list, PinPoint8/10**6)

PinPoint11 = Pres_distrib_in_Time[:, points_coords[11][0], points_coords[11][1]]
axs[2, 0].set_title(str((round(points_r_fi_pair[11][0], 2), round(points_r_fi_pair[11][1], 2))), y = 0.75, loc='left')
axs[2, 0].set_xlabel("Time, s")
axs[2, 0].set_ylabel("Pressure, MPa")
axs[2, 0].plot(t_list_exp, P_filter_11)
axs[2, 0].plot(t_list, np.array(PinPoint11)/10**6)

plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

PinPoint13 = Pres_distrib_in_Time[:, points_coords[13][0], points_coords[13][1]]
axs[0].set_title(str((round(points_r_fi_pair[13][0], 2), round(points_r_fi_pair[13][1], 2))), y = 0.75, loc='left')
axs[0].set_xlabel("Time, s")
axs[0].set_ylabel("Pressure, MPa")
axs[0].plot(t_list_exp, P_filter_13)
axs[0].plot(t_list, PinPoint13/10**6)

PinPoint12 = Pres_distrib_in_Time[:, points_coords[12][0], points_coords[12][1]]
axs[1].set_title(str((round(points_r_fi_pair[12][0], 2), round(points_r_fi_pair[12][1], 2))), y = 0.75, loc='left')
axs[1].set_xlabel("Time, s")
axs[1].set_ylabel("Pressure, MPa")
axs[1].plot(t_list_exp, P_filter_12)
axs[1].plot(t_list, PinPoint12/10**6)


plt.show()
