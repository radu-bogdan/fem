# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.interpolate as si

# # points = [[0, 0], [0, 2], [2, 3], [4, 0], [6, 3], [8, 2], [8, 0]];
# # points = np.array(points)
# # x = points[:,0]
# # y = points[:,1]

# total = 50

# x = np.linspace(1,100,total)
# y = np.random.randint(1,100,total)

# points = np.c_[x,y]

# t = range(len(points))
# ipl_t = np.linspace(0.0, len(points) - 1, 1000)

# x_tup = si.splrep(t, x, k=3)
# y_tup = si.splrep(t, y, k=3)

# x_list = list(x_tup)
# xl = x.tolist()
# x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

# y_list = list(y_tup)
# yl = y.tolist()
# y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

# x_i = si.splev(ipl_t, x_list)
# y_i = si.splev(ipl_t, y_list)

# #==============================================================================
# # Plot
# #==============================================================================

# fig = plt.figure()

# ax = fig.add_subplot(231)
# plt.plot(t, x, '-og')
# plt.plot(ipl_t, x_i, 'r')
# plt.xlim([0.0, max(t)])
# plt.title('Splined x(t)')

# ax = fig.add_subplot(232)
# plt.plot(t, y, '-og')
# plt.plot(ipl_t, y_i, 'r')
# plt.xlim([0.0, max(t)])
# plt.title('Splined y(t)')

# ax = fig.add_subplot(233)
# plt.plot(x, y, '-og')
# plt.plot(x_i, y_i, 'r')
# plt.xlim([min(x) - 0.3, max(x) + 0.3])
# plt.ylim([min(y) - 0.3, max(y) + 0.3])
# plt.title('Splined f(x(t), y(t))')

# ax = fig.add_subplot(234)
# for i in range(7):
#     vec = np.zeros(11)
#     vec[i] = 1.0
#     x_list = list(x_tup)
#     x_list[1] = vec.tolist()
#     x_i = si.splev(ipl_t, x_list)
#     plt.plot(ipl_t, x_i)
# plt.xlim([0.0, max(t)])
# plt.title('Basis splines')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from nonlinLaws import g_nonlinear_all
from nonlinLaws_bosch import g_nonlinear,gx_nonlinear,gy_nonlinear,gxx_nonlinear,gyy_nonlinear,gxy_nonlinear,gyx_nonlinear,mu,mux,muy

a = 1e7
l = 100_000
x = np.linspace(0.01,a,l)
y = np.linspace(0.01,a,l)


g_l, gx_l, gy_l, gxx_l, gxy_l, gyx_l, gyy_l,\
       g_nl,gx_nl,gy_nl,gxx_nl,gxy_nl,gyx_nl,gyy_nl = g_nonlinear_all(x,y)


# plt.loglog(g_nl,'--')
# plt.loglog(g_nonlinear(x,x))

fig, axs = plt.subplots(4, 2)

axs[0,0].loglog(g_nl,'--')
axs[0,0].loglog(g_nonlinear(x,y),'-')
# axs[0,0].set_aspect('equal', 'box')


# axs[0,1].loglog(g_nl,'--')
axs[0,1].loglog(mu(x,y),'-')
# axs[0,1].loglog(mux(x,y),'--')


axs[1,0].loglog(gx_nl,'--')
axs[1,0].loglog(gx_nonlinear(x,y),'-')
# axs[1,0].set_aspect('equal', 'box')

axs[1,1].loglog(gy_nl,'--')
axs[1,1].loglog(gy_nonlinear(x,y),'-')
# axs[1,1].set_aspect('equal', 'box')

axs[2,0].loglog(gxx_nl,'--')
axs[2,0].loglog(gxx_nonlinear(x,y),'-')
# axs[2,0].set_aspect('equal', 'box')

axs[2,1].loglog(-gxy_nl,'--')
axs[2,1].loglog(-gxy_nonlinear(x,y),'-')
# axs[2,1].set_aspect('equal', 'box')

axs[3,0].loglog(-gyx_nl,'--')
axs[3,0].loglog(-gyx_nonlinear(x,y),'-')
# axs[3,0].set_aspect('equal', 'box')

axs[3,1].loglog(gyy_nl,'--')
axs[3,1].loglog(gyy_nonlinear(x,y),'-')
# axs[3,1].set_aspect('equal', 'box')

H_KL = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 500000] + [(zz-2.46158548)/(4*np.pi*10**-7)+500000 for zz in np.linspace(2.46158548+(2.46158548-2.08458619), 10, 60)])
B_KL = np.array([0,  0.076361010000000,   0.151802010000000,   0.224483020000000,   0.294404020000000,   0.360645030000000,   0.422286030000000,   0.479327040000000,   0.532688040000000,   0.581449050000000,   0.626530050000000,   0.913580110000000,    1.047910160000000,   1.123360210000000,   1.172130270000000,   1.205260320000000,   1.231030370000000,  1.250360420000000,   1.266010480000000,   1.279820530000000,   1.356281060000000,   1.400541590000000,   1.436522120000000,   1.467902650000000,   1.497443190000000,   1.525143720000000,   1.551004250000000,   1.575944780000000,   1.599045310000000,   1.761970620000000,   1.836575930000000,   1.870701240000000,   1.891026550000000,   1.905831860000000,   1.919717170000000,   1.932682480000000,   1.945647790000000,   1.958613100000000,   2.084586190000000,   2.461585480000000] + [ zz for zz in np.linspace(2.46158548+(2.46158548-2.08458619), 10, 60)])

mu_KL = B_KL[1:]/H_KL[1:]
xx = H_KL[1:]**2

# axs[1,0].loglog(xx, mu_KL,'--')