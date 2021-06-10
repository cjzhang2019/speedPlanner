"""
Created on Wed May 13 10:24:24 2020
using Hybrid A* solve planning problem
@author: cjzhang
"""

import numpy as np
import vehicle
import collision_check
import rs_path
import grid_a_star
import math
import queue
import scipy.spatial
import matplotlib.pyplot as plt
import time
import copy
import CubicSpline

font1 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size'   : 20}
predictTime = 30
predictLength = 40
safeDistance = 0
obstaclePredictionResultList = []
obstaclePredictionResult = {}
speedLowerLimit = 0
speedUpperLimit = 40

XY_GRID_RESOLUTION = 1.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
GOAL_TYAW_TH = np.deg2rad(5.0)  # [rad]
MOTION_RESOLUTION = 1  # [m] path interporate resolution
N_STEER = 4.0  # number of steer command
EXTEND_AREA = 0.0  # [m] map extend length
SKIP_COLLISION_CHECK = 4  # skip number for collision check

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 1.0  # Heuristic cost

WB = vehicle.WB  # [m] Wheel base
MAX_STEER = vehicle.MAX_STEER  # [rad] maximum steering angle

class Node(object):

    def __init__(self, xind, yind, yawind, direction, x, y, yaw, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        #steer input
        self.cost = cost
        self.pind = pind
        # pind::Int64  # parent index

class Path(object):

    # x::Array{Float64} # x position [m]
    # y::Array{Float64} # y position [m]
    # yaw::Array{Float64} # yaw angle [rad]
    # yaw1::Array{Float64} # trailer angle [rad]
    # direction::Array{Bool} # direction forward: true, back false
    # cost::Float64 # cost

    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost

class Config(object):

    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw, xw, yw, yaww, xyreso, yawreso):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso

    def prn_obj(obj):
        print ('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

def calc_config(ox, oy, xyreso, yawreso):
    #computer capacity

    min_x_m = min(ox) - EXTEND_AREA
    min_y_m = min(oy) - EXTEND_AREA
    max_x_m = max(ox) + EXTEND_AREA
    max_y_m = max(oy) + EXTEND_AREA

    ox.append(min_x_m)
    oy.append(min_y_m)
    ox.append(max_x_m)
    oy.append(max_y_m)

    minx = round(min_x_m/xyreso)
    miny = round(min_y_m/xyreso)
    maxx = round(max_x_m/xyreso)
    maxy = round(max_y_m/xyreso)

    xw = round(maxx - minx)
    yw = round(maxy - miny)

    minyaw = 0
    maxyaw = round(math.pi/(2 * yawreso))
    yaww = round(maxyaw - minyaw)

    # minyawt = minyaw
    # maxyawt = maxyaw
    # yawtw = yaww

    afterConfig = Config(minx, miny, minyaw, maxx, maxy, maxyaw, xw, yw, yaww, xyreso, yawreso)
    return afterConfig

def calc_index(node, c):
    ind = (node.yawind - c.minyaw)*c.xw*c.yw+(node.yind - c.miny)*c.xw + (node.xind - c.minx)
    # 3D grid
    if ind <= 0:
        print ("Error(calc_index):", ind)

    return ind

def calc_holonomic_with_obstacle_heuristic(Node, ox , oy , xyreso):
    h_dp = grid_a_star.calc_dist_policy(Node.x[-1], Node.y[-1], ox, oy, xyreso, 1.0)
    return h_dp

def calc_cost(n, h_dp, c):

   return (n.cost + 3 * H_COST * h_dp[int(n.xind - c.minx)][int(n.yind - c.miny)])

def calc_motion_inputs():
    up = []
    #0-N_STEER-1
    for i in range(int(N_STEER)-1,-1,-1):
        x = MAX_STEER - i*(MAX_STEER/N_STEER)
        up.append(x)

    u = [0.0] + [i for i in up] + [-i for i in up]
    d = [1.0 for i in range(len(u))]
    #d = 1 move forward d = -1 move backward
    print(u)
    print(d)
    return u, d

def calc_rs_path_cost(rspath):
    #length + forward backward switch + steer + steer switch

    cost = 0.0
    for l in rspath.lengths:
        if l >= 0:  # forward
            cost += l
        else:  # back
            cost += abs(l) * BACK_COST

    # switch back penalty
    for i in range(len(rspath.lengths)-1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:  # switch back
            cost += SB_COST

    # steer penalyty
    for ctype in rspath.ctypes:
        if ctype != "S" : # curve
            cost += STEER_COST * abs(MAX_STEER)

    # steer switch profile
    nctypes = len(rspath.ctypes)
    ulist = [0.0 for i in range(nctypes)]
    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = - MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes)-1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost

def analystic_expantion(n, ngoal, ox, oy, kdtree):
    #here I corrected the mistake

    sx = n.x[-1]
    sy = n.y[-1]
    syaw = n.yaw[-1]

    max_curvature = math.tan(MAX_STEER)/WB
    paths = rs_path.calc_paths(sx, sy, syaw, ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1], max_curvature, step_size = MOTION_RESOLUTION)

    if len(paths) == 0:
        return None

    pathset = {}
    path_id = 0
    for path in paths:
        pathset[path_id] = path
        path_id = path_id + 1

    p_idList = sorted(pathset, key=lambda x: calc_rs_path_cost(pathset[x]))
    for i in p_idList:
        path = pathset[i]
        if collision_check.check_collision(ox, oy, path.x, path.y, path.yaw, kdtree):
            return path # path is ok

    return None

def update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree):

    apath = analystic_expantion(current, ngoal, ox, oy, kdtree)
    if apath != None:
        fx = apath.x[1:]
        fy = apath.y[1:]
        fyaw =  apath.yaw[1:]
        fcost = current.cost + calc_rs_path_cost(apath)
        fpind = calc_index(current, c)

        fd = []
        for d in apath.directions[1:]:
            if d >= 0:
                fd.append(True)
            else:
                fd.append(False)
        fsteer = 0.0
        fpath = Node(current.xind, current.yind, current.yawind, current.direction, fx, fy, fyaw, fd, fsteer, fcost, fpind)
        return True, fpath
    return False, None #no update

def calc_next_node(current, c_id, u, d, c):

    arc_l = XY_GRID_RESOLUTION * 1.5

    nlist = math.ceil(arc_l / MOTION_RESOLUTION) + 1

    xlist, ylist, yawlist = [], [], []

    xlist_0 = current.x[-1] + d * MOTION_RESOLUTION * math.cos(current.yaw[-1])
    ylist_0 = current.y[-1] + d * MOTION_RESOLUTION * math.sin(current.yaw[-1])
    yawlist_0 = rs_path.pi_2_pi(current.yaw[-1] + d * MOTION_RESOLUTION / WB * math.tan(u))
    xlist.append(xlist_0)
    ylist.append(ylist_0)
    yawlist.append(yawlist_0)

    for i in range(1,int(nlist)):
        xlist_i = xlist[i-1] + d * MOTION_RESOLUTION * math.cos(yawlist[i-1])
        ylist_i = ylist[i-1] + d * MOTION_RESOLUTION * math.sin(yawlist[i-1])
        yawlist_i = rs_path.pi_2_pi(yawlist[i-1] + d * MOTION_RESOLUTION / WB * math.tan(u))
        xlist.append(xlist_i)
        ylist.append(ylist_i)
        yawlist.append(yawlist_i)

    xind = round(xlist[-1] / c.xyreso)
    yind = round(ylist[-1] / c.xyreso)
    yawind = round(yawlist[-1] / c.yawreso)

    addedcost = 0.0
    if d > 0:
        direction = True
        addedcost += abs(arc_l)
    else:
        direction = False
        addedcost += abs(arc_l) * BACK_COST

    # swich back penalty
    if direction != current.direction:  # switch back penalty
        addedcost += SB_COST

    # steer penalyty
    addedcost += STEER_COST * abs(u)

    # steer change penalty
    addedcost += STEER_CHANGE_COST * abs(current.steer - u)

    cost = current.cost + addedcost

    directions = [direction for i in range(len(xlist))]
    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist, directions, u, cost, c_id)

    return node

def verify_index(node, c, ox, oy, kdtree):

    # overflow map
    if (node.xind - c.minx) >= c.xw:
        return False
    elif (node.xind - c.minx) <= 0:
        return False
    if (node.yind - c.miny) >= c.yw:
        return False
    elif (node.yind - c.miny) <= 0:
        return False

    if collision_check.check_collision(ox, oy, node.x, node.y, node.yaw, kdtree) == False:
        return False
    
    for i in range(len(node.x)):
        if node.x[i] < node.x[0]:
            return False
    for i in range(len(node.yaw)):
        if node.yaw[i] > np.deg2rad(80.0):
            return False
        if node.yaw[i] < np.deg2rad(20.0):
            return False
    
    return True #index is ok"

def is_same_grid(node1, node2):

    if node1.xind != node2.xind:
        return False
    if node1.yind != node2.yind:
        return False
    if node1.yawind != node2.yawind:
        return False
    return True

#def get_final_path(closed, ngoal, nstart, c, last_A_star_node_ind):
#    
#    rx1, ry1, ryaw1 = ngoal.x[::-1], ngoal.y[::-1], ngoal.yaw[::-1]
#    
#    direction1 = ngoal.directions[::-1]
#    finalcost1 = ngoal.cost
#    nid = last_A_star_node_ind
#    rx1.extend(closed[last_A_star_node_ind].x[::-1])
#    ry1.extend(closed[last_A_star_node_ind].y[::-1])
#    ryaw1.extend(closed[last_A_star_node_ind].yaw[::-1])
#    direction1.extend(closed[last_A_star_node_ind].directions[::-1])
#    path1 = Path(rx1, ry1, ryaw1, direction1, finalcost1)
#
#    
#    rx2, ry2, ryaw2, direction2, finalcost2 = [], [], [], [], []
#    finalcost2 = closed[nid].cost
#
#    while 1:
#        n = closed[nid]
#        rx2.extend(n.x[::-1])
#        ry2.extend(n.y[::-1])
#        ryaw2.extend(n.yaw[::-1])
#        direction2.extend(n.directions[::-1])
#        nid = n.pind
#        if is_same_grid(n, nstart):
#            break
#    path2 = Path(rx2, ry2, ryaw2, direction2, finalcost2)
#    for i in range(len(rx2)):
#        rx1.append(rx2[i])
#        ry1.append(ry2[i])
#        ryaw1.append(ryaw2[i])
#        direction1.append(direction2[i])
#    finalcost = finalcost1 + finalcost2
#    path = Path(rx1, ry1, ryaw1, direction1, finalcost)
#    
#    
#    return path1, path2, path
def get_final_path(closed, ngoal, nstart, c, last_A_star_node_ind):

    rx, ry, ryaw = ngoal.x[::-1], ngoal.y[::-1], ngoal.yaw[::-1]
    direction = ngoal.directions[::-1]
    nid = last_A_star_node_ind
    finalcost = ngoal.cost
#    if(len(rx) == 1):
#        rx.append(closed[nid].x[-1])
#        ry.append(closed[nid].y[-1])
#        ryaw.append(closed[nid].yaw[-1])
#        direction.append(closed[nid].directions[-1])
#        finalcost = closed[nid].cost
    finalcost1 = finalcost - closed[nid].cost
    rx1 = copy.deepcopy(rx)
    ry1 = copy.deepcopy(ry)
    ryaw1 = copy.deepcopy(ryaw)
    direction1 = copy.deepcopy(direction)
    rx1 = rx1[::-1]
    ry1 = ry1[::-1]
    ryaw1 = ryaw1[::-1]
    direction1 = direction1[::-1]
    path1 = Path(rx1, ry1, ryaw1, direction1, finalcost1)

    while 1:
        n = closed[nid]
        rx.extend(n.x[::-1])
        ry.extend(n.y[::-1])
        ryaw.extend(n.yaw[::-1])
        direction.extend(n.directions[::-1])
        nid = n.pind
        if is_same_grid(n, nstart):
            break
    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direction = direction[::-1]

    # adjuct first direction
    direction[0] = direction[1]

    path = Path(rx, ry, ryaw, direction, finalcost)
    

    return path1, path

class KDTree:
    """
    Nearest neighbor search class with KDTree
    Dimension is two
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        k=1 means to query the nearest neighbours and return squeezed result
        inp: input data
        """

        if len(inp.shape) >= 2:  # multi input 
            index = []
            dist = []
            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist
        else:
            dist, index = self.tree.query(inp, k=k)
            return index, dist

    def search_in_distance(self, inp, r):
        """
        find points within a distance r
        """
        index = self.tree.query_ball_point(inp, r)
        return index

def calc_hybrid_astar_path(sx , sy , syaw , gx , gy , gyaw ,  ox , oy , xyreso , yawreso):

    # sx: start x position[m]
    # sy: start y position[m]
    # gx: goal x position[m]
    # gy: goal y position[m]
    # ox: x position list of Obstacles[m]
    # oy: y position list of Obstacles[m]
    # xyreso: grid resolution[m]
    # yawreso: yaw angle resolution[rad]

    syaw0 = rs_path.pi_2_pi(syaw)
    gyaw0 = rs_path.pi_2_pi(gyaw)
    #keep -pi-pi
    global tox,toy
    ox, oy = ox[:], oy[:]
    tox, toy = ox[:], oy[:]
    kdtree = KDTree(np.vstack((tox, toy)).T)
    #use kdtree to represent obstacles logN < N

    c = calc_config(ox, oy, xyreso, yawreso)
    nstart = Node(round(sx / xyreso), round(sy / xyreso), round(syaw0 / yawreso), True, [sx], [sy], [syaw0], [True], 0.0, 0.0, -1)
    ngoal = Node(round(gx/xyreso), round(gy/xyreso), round(gyaw0/yawreso), True, [gx], [gy], [gyaw0], [True], 0.0, 0.0, -1)
    h_dp = calc_holonomic_with_obstacle_heuristic(ngoal, ox, oy, xyreso)
    
    #cost from the goal to each point index
    openset, closedset = {},{}
    fnode = ngoal
    openset[calc_index(nstart, c)] = nstart

    u, d = calc_motion_inputs()
    nmotion = len(u)

    if collision_check.check_collision(ox, oy, [sx], [sy], [syaw0], kdtree) == False:
        return [], []
    if collision_check.check_collision(ox, oy, [gx], [gy], [gyaw0], kdtree) == False:
        return [], []
    times = 0
    last_A_star_node_ind = 0
    

    while 1:
#        if times > 1000:
#            return [],[]
        if len(openset) == 0:
            print ("Error: Cannot find path, No open set")
            return []

        c_id = min(openset, key=lambda o: calc_cost(openset[o], h_dp, c))

        current = openset[c_id]

        # move current node from open to closed
        del openset[c_id]
        closedset[c_id] = current
        plt.figure(2)
        plt.plot(current.x[::-1],current.y[::-1],"rx")
        
        if ((current.x[-1] - gx) ** 2 + (current.y[-1] - gy) ** 2  < 2 or current.x[-1] > 20):
            print("currentx:", current.x[-1])
            last_A_star_node_ind = c_id
            break

#        isupdated, fpath = update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree)
#        if isupdated:  # found
#            last_A_star_node_ind = c_id
#            fnode = fpath
#            break


        for i in range(nmotion):
            node = calc_next_node(current, c_id, u[i], d[i], c)

            if verify_index(node, c, ox, oy, kdtree) == False:
                continue

            node_ind = calc_index(node, c)

            # If it is already in the closed set, skip it
            if node_ind in closedset:
                continue

            if node_ind not in openset:
                openset[node_ind] = node
                
            else:
                if calc_cost(openset[node_ind], h_dp, c) > calc_cost(node, h_dp, c):
                    # If so, update the node to have a new parent
                    openset[node_ind] = node
        times = times + 1
    path1, path = get_final_path(closedset, fnode, nstart, c, last_A_star_node_ind)
    plt.show()

    return path1, path
        
def generate_target_course(x, y):
    csp = CubicSpline.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))
    return rx, ry, ryaw, rk, csp

def obstaclePrediction(obList, tx, ty, predictTime, predictLength):
    #here I need to get t1,t2,s1,s2 of every obstacle
    x0 = []
    y0 = []
    s = []
    d = []
    t1 = []
    t2 = []
    s1 = []
    s2= []
    s3 = []
    s4 = []
    for i in range(len(obList)):
        x0.append(0)
        y0.append(0)
        s.append(0)
        d.append(0)
        t1.append(0)
        t2.append(0)
        s1.append(0)
        s2.append(0)
        s3.append(0)
        s4.append(0)
    for obNumber in range(len(obList)): 
        for i in range(len(tx)-1):
            a = [math.cos(math.pi*obList[obNumber]['heading']/180), math.sin(math.pi*obList[obNumber]['heading']/180)]
            b = [tx[i]-obList[obNumber]['x'],ty[i]-obList[obNumber]['y']]
            c = [tx[i+1]-obList[obNumber]['x'],ty[i+1]-obList[obNumber]['y']]
            if((a[0]*b[1]-a[1]*b[0])*(a[0]*c[1]-a[1]*c[0]) > 0 or (a[0]*b[0]+a[1]*b[1]) < 0):
                s[obNumber] = s[obNumber] + math.sqrt((tx[i+1] - tx[i]) * (tx[i+1] - tx[i]) + (ty[i+1] - ty[i]) * (ty[i+1] - ty[i]))
            else:
                if(obList[obNumber]['heading'] == 90 or obList[obNumber]['heading'] == -90):
                    x0[obNumber] = obList[obNumber]['x']
                    y0[obNumber] = (ty[i+1]-ty[i])/(tx[i+1]-tx[i])*(x0[obNumber]-tx[i])+ty[i]
                    s[obNumber] = s[obNumber] + math.sqrt((x0[obNumber] - tx[i]) * (x0[obNumber] - tx[i]) + (y0[obNumber] - ty[i]) * (y0[obNumber] - ty[i]))
                else:
                    if(tx[i] == tx[i+1]):
                        x0[obNumber] = tx[i]
                        y0[obNumber] = math.tan(math.pi*obList[obNumber]['heading']/180) * (x0[obNumber]-obList[obNumber]['x']) + obList[obNumber]['y']
                        s[obNumber] = s[obNumber] + math.sqrt((x0[obNumber] - tx[i]) * (x0[obNumber] - tx[i]) + (y0[obNumber] - ty[i]) * (y0[obNumber] - ty[i]))
                    else:  
                        x0[obNumber] = ((math.tan(math.pi*obList[obNumber]['heading']/180)) * obList[obNumber]['x'] + ty[i] - obList[obNumber]['y'] - (ty[i+1]-ty[i])/(tx[i+1]-tx[i]) * tx[i])/((math.tan(math.pi*obList[obNumber]['heading']/180))-(ty[i+1]-ty[i])/(tx[i+1]-tx[i]))
                        y0[obNumber] = math.tan(math.pi*obList[obNumber]['heading']/180) * (x0[obNumber]-obList[obNumber]['x']) + obList[obNumber]['y']
                        s[obNumber] = s[obNumber] + math.sqrt((x0[obNumber] - tx[i]) * (x0[obNumber] - tx[i]) + (y0[obNumber] - ty[i]) * (y0[obNumber] - ty[i]))
                break
    for obNumber in range(len(obList)):
        if(x0[obNumber] == 0 and y0[obNumber] == 0):
            obstaclePredictionResult = {'id':obList[obNumber]['id'],'pointX':[],'pointY':[]}
            obstaclePredictionResultList.append(obstaclePredictionResult)
        else:
            d[obNumber] = math.sqrt((obList[obNumber]['x'] - x0[obNumber]) * (obList[obNumber]['x'] - x0[obNumber]) + (obList[obNumber]['y'] - y0[obNumber]) * (obList[obNumber]['y'] - y0[obNumber]))
            t1[obNumber] = (d[obNumber])/obList[obNumber]['speed']
            t2[obNumber] = t1[obNumber] + 0.4
            #这里需要好好思考一下，我设置的是预测她在我们道路上10s，但事实应该是会一直在？
            s1[obNumber] = s[obNumber]
            s2[obNumber] = obList[obNumber]['speed']*(t2[obNumber] - t1[obNumber]) + s1[obNumber]
            s3[obNumber] = s2[obNumber] + 5
            s4[obNumber] = s1[obNumber] + 5
            obstaclePredictionResult = {'id':obList[obNumber]['id'],'pointX':[t1[obNumber], t2[obNumber], t2[obNumber], t1[obNumber]],'pointY':[s1[obNumber], s2[obNumber], s3[obNumber], s4[obNumber]],'sDecision':obList[obNumber]['sDecision']}
            obstaclePredictionResultList.append(obstaclePredictionResult)
    return obstaclePredictionResultList

def drawPerceptionResult(obList, tx, ty):
    plt.figure(1)
    plt.plot(tx, ty, "r",label = 'path')
    for obNumber in range(len(obList)):
        plt.plot(obList[obNumber]['x'],obList[obNumber]['y'] , "*", label = ('obstacle', obList[obNumber]['sDecision']))
        plt.quiver(obList[obNumber]['x'], obList[obNumber]['y'], math.cos(math.pi*obList[obNumber]['heading']/180), math.sin(math.pi*obList[obNumber]['heading']/180), color='b', width=0.005)
        
    plt.axis('equal')
    plt.xlabel('x(m)',font1)
    plt.ylabel('y(m)',font1)
    plt.legend()
    
def drawObstaclePredictionResult(obstaclePredictionResultList):
    plt.figure(2)
    plt.xlim(0,predictTime)
    plt.ylim(0,predictLength)
    plt.xlabel('t(s)',font1)
    plt.ylabel('s(m)',font1)
    for obNumber in range(len(obstaclePredictionResultList)):
        plt.fill(obstaclePredictionResultList[obNumber]['pointX'],obstaclePredictionResultList[obNumber]['pointY'], facecolor='r')
    plt.title('Obstacle Prediction Result')
    plt.show()

def gridMapEstablished(obstaclePredictionResultList, safeDistance):
    ##建图并设立障碍
    ox = []
    oy = []
    obstacle = []
    obstacle.append((-5,-5))
    obstacle.append((35,-5))
    obstacle.append((35,45))
    obstacle.append((-5,45))
    for (x,y) in obstacle:
        ox.append(x)
        oy.append(y)
    for obNumber in range(len(obstaclePredictionResultList)):
        x_w1 = obstaclePredictionResultList[obNumber]['pointX'][0] * 10
        y_w1 = obstaclePredictionResultList[obNumber]['pointY'][0] - safeDistance
        x_w2 = obstaclePredictionResultList[obNumber]['pointX'][1] * 10
        y_w2 = obstaclePredictionResultList[obNumber]['pointY'][1] - safeDistance
        x_w3 = obstaclePredictionResultList[obNumber]['pointX'][2] * 10
        y_w3 = obstaclePredictionResultList[obNumber]['pointY'][2] + safeDistance
        x_w4 = obstaclePredictionResultList[obNumber]['pointX'][3] * 10
        y_w4 = obstaclePredictionResultList[obNumber]['pointY'][3] + safeDistance
        obX = [x_w1, x_w2, x_w3, x_w4, x_w1]
        obY = [y_w1, y_w2, y_w3, y_w4, y_w1]
        if obstaclePredictionResultList[obNumber]['sDecision'] == 0:
            for i in range(math.floor(x_w1), math.ceil(x_w2 + 1), 1):
                for j in range(math.floor(y_w1), predictLength + 1, 1):
                    if((i == x_w1 and j == y_w1) or (i == x_w2 and j == y_w2) or (i == x_w3 and j == y_w3) or (i == x_w4 and j == y_w4)):
                        ox.append(i)
                        oy.append(j)
                    elif (x_w2 - x_w1) * (j - y_w1) - (y_w2 - y_w1) * (i - x_w1) > 0:
                        ox.append(i)
                        oy.append(j)
#                    else:
#                        sumangle = 0.0
#                        for k in range(len(obX)-1):
#                            x1 = obX[k] - i
#                            y1 = obY[k] - j
#                            x2 = obX[k+1] - i
#                            y2 = obY[k+1] - j
#                            d1 = math.hypot(x1,y1)
#                            d2 = math.hypot(x2,y2)
#                            tmp = (x1 * x2 + y1 * y2) / (d1 * d2)
#                            sumangle += math.acos(tmp)
#                
#                        if abs(sumangle - 2 * math.pi) < 0.001:
#                            ox.append(i)
#                            oy.append(j)
        if obstaclePredictionResultList[obNumber]['sDecision'] == 1:
            for i in range(math.floor(x_w1), math.ceil(x_w2 + 1), 1):
                for j in range(0, math.ceil(y_w3 + 1), 1):
                    if((i == x_w1 and j == y_w1) or (i == x_w2 and j == y_w2) or (i == x_w3 and j == y_w3) or (i == x_w4 and j == y_w4)):
                        ox.append(i)
                        oy.append(j)
                    elif (x_w3 - x_w4) * (j - y_w4) - (y_w3 - y_w4) * (i - x_w4) < 0:
                        ox.append(i)
                        oy.append(j)
#                    else:
#                        sumangle = 0.0
#                        for k in range(len(obX)-1):
#                            x1 = obX[k] - i
#                            y1 = obY[k] - j
#                            x2 = obX[k+1] - i
#                            y2 = obY[k+1] - j
#                            d1 = math.hypot(x1,y1)
#                            d2 = math.hypot(x2,y2)
#                            tmp = (x1 * x2 + y1 * y2) / (d1 * d2)
#                            sumangle += math.acos(tmp)
#                
#                        if abs(sumangle - 2 * math.pi) < 0.001:
#                            ox.append(i)
#                            oy.append(j)
                                  
    return ox,oy

def drawVelocityPlanningResult(velocityPointX, velocityPointY, obstaclePredictionResultList):
    plt.figure(3)
    plt.xlim(0,predictTime)
    plt.ylim(0,predictLength)

    for obNumber in range(len(obstaclePredictionResultList)):
        x_w1 = obstaclePredictionResultList[obNumber]['pointX'][0] * 10
        x_w2 = obstaclePredictionResultList[obNumber]['pointX'][1] * 10
        x_w3 = obstaclePredictionResultList[obNumber]['pointX'][2] * 10
        x_w4 = obstaclePredictionResultList[obNumber]['pointX'][3] * 10
        a = [x_w1,x_w2,x_w3,x_w4]
        plt.fill(a,obstaclePredictionResultList[obNumber]['pointY'],label = ('obstacle',obstaclePredictionResultList[obNumber]['sDecision']))
    plt.xlabel('10 * t/(s)',font1)
    plt.ylabel('s/(m)',font1)
    plt.plot(velocityPointX, velocityPointY,'r')
    plt.title('Speed Planning Result', font1)
    plt.legend()
    plt.show()

def main():
    obList = []
    wx = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0]
    wy = [0.0, 0.0, 0.0, -1, -2, -3, -4, -4, -4, -4, -4]
    ob1 = {'id':1,'x':6,'y':-4,'speed':15,'heading':0, 'sDecision':0}
    ob2 = {'id':2,'x':3,'y':3,'speed':8,'heading':-30, 'sDecision':1}
#    ob3 = {'id':3,'x':50,'y':12,'speed':1,'heading':-30, 'sDecision':0}
#here is the output of decision module, 0 means following, 1 means overtaking
    obList.append(ob1)
    obList.append(ob2)
#    obList.append(ob3)
    tx, ty, tyaw, tk, csp = generate_target_course(wx, wy)
    obstaclePredictionResultList = obstaclePrediction(obList, tx, ty, predictTime, predictLength)
    drawPerceptionResult(obList, tx, ty)
    
#    drawObstaclePredictionResult(obstaclePredictionResultList)
    ox, oy = gridMapEstablished(obstaclePredictionResultList, safeDistance)
    plt.figure(2)
    plt.plot(ox,oy,'o')
    time1 = time.time()
    PATH1, PATH = calc_hybrid_astar_path(0 , 0 , np.deg2rad(45) , 28 , 38 , np.deg2rad(45) ,  ox , oy , XY_GRID_RESOLUTION , YAW_GRID_RESOLUTION)
    time2 = time.time()
    t = time2 - time1
    print("during time: ",t)
    velocityPointX = []
    velocityPointY = []
    velocityV = []
    if PATH1 == [] and PATH == []:
        print ('Error: Cannot find path, overtime')
    else:
        velocityPointX = PATH.x
        velocityPointY = PATH.y
        for i in range(len(velocityPointX)-1):
            velocityV.append(10*(velocityPointY[i+1] - velocityPointY[i])/(velocityPointX[i+1] - velocityPointX[i]))
    drawVelocityPlanningResult(velocityPointX, velocityPointY, obstaclePredictionResultList)
    print(velocityV)
    
    
    
if __name__ == '__main__':
    main()