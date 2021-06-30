import numpy as np
import math
import cvxopt
import matplotlib.pyplot as plt

v_max = 13.8889        # velocity limit [m/s]
a_max = 2.0            # acceleration limit [m/s2]
s_max = 3.0            # jerk limit [m/s3]
latacc_max = 2.0       # lateral acceleration limit [m/s2]
tire_angvel_max = 0.7  # tire angular velocity max [rad/s] (calculated with kinematics model)
tire_angvel_thr = 0.1  # Threshold to judge that the tire has the angular velocity [rad/s]
vel_min_for_tire = 2.0 # Minimum vehicle speed when moving a tire [m]
waypoints_dist = 1.0  # distance between each waypoints [m]
wheelbase = 2.9       # [m]
# -- max iteration number for convex optimization --
max_iter_num = 11

def area2(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx**2 + dy**2)**0.5

def calc_3points_curvature(a, b, c):
    return 2.0 * area2(a, b, c) / (dist(a,b) * dist(b,c) * dist(c,a))

# curvature calculation
def calc_waypoints_curvature(wp_x, wp_y):
    l = len(wp_x)
    curvature = np.zeros((l,1))
    for i in range(1,l-1):
        p0 = np.array([wp_x[i-1], wp_y[i-1]])
        p1 = np.array([wp_x[i], wp_y[i]])
        p2 = np.array([wp_x[i+1], wp_y[i+1]])
        curvature[i] = calc_3points_curvature(p0, p1, p2)
    curvature[0] = curvature[1]
    curvature[l-1] = curvature[l-2]
    return curvature

def convert_eluer_to_monotonic(yaw_arr):
    l = len(yaw_arr)
    for i in range(1, l):
        if yaw_arr[i] - yaw_arr[i-1] > math.pi:
            yaw_arr[i:l] = yaw_arr[i:l] - 2.0 * math.pi
        elif yaw_arr[i] - yaw_arr[i-1] < -math.pi:
            yaw_arr[i:l] = yaw_arr[i:l] + 2.0 * math.pi
    return yaw_arr

# optimization
def plan_speed_convex_opt(vel, v_max_arr, v_min_arr, yaw):

    cvxopt.solvers.options['abstol'] = 1e-15
    cvxopt.solvers.options['reltol'] = 1e-15
    cvxopt.solvers.options['feastol'] = 1e-15

    l = len(vel)

    # initial condition & final condition as a constaraint
    A = np.zeros((2,l))
    A[0,0] = 1
    A[1,l-1] = 1
    b = np.array([vel[0], vel[l-1]])

    for j in range(max_iter_num):

        # velocity constraint
        G_vel = np.eye(l)
        G_vel = np.vstack((G_vel, -G_vel))
        h_vel_max = v_max_arr # velocity limit for lateral acceleration 
        h_vel_min = -v_min_arr
        h_vel = np.vstack((h_vel_max, h_vel_min))
        G = G_vel
        h = h_vel

        # acceleration constraint
        G_acc = np.zeros((l-1, l))
        for i in range(l-1):
            G_acc[i,i] = -vel[i] / waypoints_dist
            G_acc[i,i+1] = vel[i] / waypoints_dist
        G_acc = np.vstack((G_acc, -G_acc))
        h_acc = np.ones((l-1,1)) * a_max
        h_acc = np.vstack((h_acc, h_acc))
        G = np.vstack((G, G_acc))
        h = np.vstack((h, h_acc))

        # jerk constraint
        G_jerk = np.zeros((l-2, l))
        for i in range(l-2):
            G_jerk[i,i] = (vel[i+1] / waypoints_dist)**2
            G_jerk[i,i+1] = -2.0 * ((vel[i+1] / waypoints_dist)**2)
            G_jerk[i,i+2] = (vel[i+1] / waypoints_dist)**2
        G_jerk = np.vstack((G_jerk, -G_jerk))
        h_jerk = np.ones((l-2,1)) * s_max
        h_jerk = np.vstack((h_jerk, h_jerk))
        G = np.vstack((G, G_jerk))
        h = np.vstack((h, h_jerk))


        # tire angvel constraint
        G_tire = np.zeros((l-2, l))
        for i in range(l-2):
            G_tire[i,i+1] = wheelbase * (yaw[i+2] - 2.0*yaw[i+1] + yaw[i]) / (waypoints_dist**2)
        G_tire = np.vstack((G_tire, -G_tire))
        h_tire = np.ones((l-2,1)) * tire_angvel_max
        h_tire = np.vstack((h_tire, h_tire))
        G = np.vstack((G, G_tire))
        h = np.vstack((h, h_tire))


        # minimize squared error from original velocity
        P = np.eye(l)
        q = np.array(-vel)

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), G=cvxopt.matrix(G), h=cvxopt.matrix(h), A=cvxopt.matrix(A), b=cvxopt.matrix(b))
        # sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), G=cvxopt.matrix(G), h=cvxopt.matrix(h))
        vel_result = np.array(sol['x'])
        cost = sol["primal objective"] + np.dot(vel, vel)

    return vel_result


def qpSolver(x, y, yaw, vel_orig):
    x = np.array(x)
    y = np.array(y)
    yaw = np.array(yaw)
    vel_orig = np.array(vel_orig)
 
    yaw = convert_eluer_to_monotonic(yaw)
    curvature = calc_waypoints_curvature(x, y)

    v = vel_orig
    l = len(v)
    acc_orig = v[1:l] * np.array(v[1:l] - v[0:l-1]) / waypoints_dist
    jerk_orig = (v[1:l-1] * v[1:l-1]) * np.array(v[2:l] - 2.0 * v[1:l-1] + v[0:l-2]) / (waypoints_dist ** 2)
    latacc_orig = np.abs(curvature) * v.reshape((l,1)) * v.reshape((l,1))
    tire_angvel_orig = v[1:l-1] * (yaw[2:l] - 2.0 * yaw[1:l-1] + yaw[0:l-2]) / (waypoints_dist ** 2) * wheelbase
    

    # -- calculate max velocity with lateral acceleration constraint --
    v_max_arr = np.zeros((l,1))
    for i in range(l):
        k = max(np.abs(curvature[i]), 0.0001) # to avoid 0 devide
        v_max_arr[i] = min((latacc_max / k) ** 0.5, v_max)
    
    # -- calculate min velocity for moving tire
    tire_angvel_tmp = np.zeros((l,1))
    for i in range(1,l-1):
        tire_angvel_tmp[i] = tire_angvel_orig[i-1]
    tire_angvel_tmp[0] = tire_angvel_orig[0]
    tire_angvel_tmp[-1] = tire_angvel_orig[-1]

    v_min_arr = np.zeros((l,1))
    for i in range(l):
        if abs(tire_angvel_tmp[i]) > tire_angvel_thr:
            v_min_arr[i] = vel_min_for_tire
    


    # -- solve optimization problem --
    vel_result = plan_speed_convex_opt(vel_orig, v_max_arr, v_min_arr, yaw)
    
    l = len(vel_result)
    v = vel_result.reshape((1,l))
    v = v[0,:]
    k = curvature.reshape((1,l))
    k = k[0,:]

    acc_res = v[0:l-1] * (v[1:l] - v[0:l-1])
    jerk_res = (v[1:l-1] * v[1:l-1]) * (v[2:l] - 2.0 * v[1:l-1] + v[0:l-2]) / (waypoints_dist ** 2)
    tire_angvel_res = v[1:l-1] * (yaw[2:l] - 2.0 * yaw[1:l-1] + yaw[0:l-2]) / (waypoints_dist ** 2) * wheelbase
    latacc_res = k* v * v

    print('-- result --')
    print('acc lim = ', a_max)
    print('acc max = ', np.max(acc_res))
    print('acc min = ', np.min(acc_res))

    print('jerk lim = ', s_max)
    print('jerk max = ', np.max(jerk_res))
    print('jerk min = ', np.min(jerk_res))

    print('latacc lim = ', latacc_max)
    print('latacc max = ', np.max(latacc_res))
    print('latacc min = ', np.min(latacc_res))
    fig = plt.figure(1, figsize=(12, 8))

    ax1 = fig.add_subplot(231)
    ax1.plot(vel_orig, '-o', label="original")
    ax1.plot(vel_result, '-o', label="optimized")
    ax1.set_title('velocity')
    ax1.legend()

    ax2 = fig.add_subplot(232)
    ax2.plot(acc_orig, '-o', label="original")
    ax2.plot(acc_res, '-o', label="optimized")
    ax2.plot(np.ones(l) * a_max, '--', label="limit")
    ax2.set_title('acceleration')
    ax2.legend()

    ax3 = fig.add_subplot(233)
    ax3.plot(jerk_orig, '-o', label="original")
    ax3.plot(jerk_res, '-o', label="optimized")
    ax3.plot(np.ones(l) * s_max, '--', label="limit")
    ax3.set_title('jerk')
    ax3.legend()

    ax4 = fig.add_subplot(234)
    ax4.plot(latacc_orig, '-o', label="original")
    ax4.plot(latacc_res, '-o', label="optimized")
    ax4.plot(np.ones(l) * latacc_max, '--', label="limit")
    ax4.set_title('lateral acceleration')
    ax4.legend()

    ax5 = fig.add_subplot(235)
    ax5.plot(tire_angvel_orig, '-o', label="original")
    ax5.plot(tire_angvel_res, '-o', label="optimized")
    ax5.plot(np.ones(l) * tire_angvel_max, '--', label="limit")
    ax5.set_title('tire angular velocity [rad/s]')
    ax5.legend()

    ax6 = fig.add_subplot(236)
    ax6.plot(vel_result, '-o', label="velocity")
    ax6.plot(tire_angvel_res, '-o', label="tire angular vel")
    ax6.plot(v_min_arr, '--', label="minimum velocity for tire move")
    ax6.set_title('velocity threshold')
    ax6.legend()

    plt.show()

