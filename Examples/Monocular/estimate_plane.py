import random
import numpy as np
from math import acos, sin, cos, fabs, sqrt, log
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation as R


def getData(filepath, row_need=1000):
    map_points = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines[1:]:
            cleaned_data = ''.join(filter(lambda x: x.isprintable(), line))
            _, x, y, z = cleaned_data.split()
            x = float(x)
            y = float(y)
            z = float(z)   
            map_points.append([x, y, z])
    return np.array(map_points)

def rotation_matrix_to_quaternion(R):
    q = np.zeros(4)
    trace = np.trace(R)

    if trace > 0:
        q[0] = np.sqrt(1 + trace) / 2
        q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
        q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
        q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            q[1] = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
            q[0] = (R[2, 1] - R[1, 2]) / (4 * q[1])
            q[2] = (R[0, 1] + R[1, 0]) / (4 * q[1])
            q[3] = (R[0, 2] + R[2, 0]) / (4 * q[1])
        elif R[1, 1] > R[2, 2]:
            q[2] = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) / 2
            q[0] = (R[0, 2] - R[2, 0]) / (4 * q[2])
            q[1] = (R[1, 0] + R[0, 1]) / (4 * q[2])
            q[3] = (R[1, 2] + R[2, 1]) / (4 * q[2])
        else:
            q[3] = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) / 2
            q[0] = (R[1, 0] - R[0, 1]) / (4 * q[3])
            q[1] = (R[0, 2] + R[2, 0]) / (4 * q[3])
            q[2] = (R[1, 2] + R[2, 1]) / (4 * q[3])

    return q

def quaternion_to_rotation_matrix(q):
    q_w, q_x, q_y, q_z = q
    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
        [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
        [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
    ])
    return R

def plane_to_se3_3(n, d):
    # 法向量单位化
    print("法向量：", n)
    n = n / np.linalg.norm(n)
    print("法向量：", n)

    # 平移向量，沿着法向量方向偏移 d 的距离
    t = d * n

    # 构建旋转矩阵 R，使得 z 轴与法向量 n 对齐
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, n) # v轴是 n 旋转到 z_axis的旋转轴
    s = np.linalg.norm(v) # v轴的长度，z_axis和n之间的正弦值
    c = np.dot(z_axis, n) # 点积，z_axis和n之间的余弦值

    print("v: ", v)
    print("s: ", s)
    print("c: ", c)
    # 罗德里格斯旋转公式
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    if s != 0:
        R_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
    else:
        # 若法向量与 z 轴平行，则直接用单位矩阵
        R_matrix = np.eye(3)
    print(R_matrix)

    # 构建 4x4 齐次变换矩阵 SE(3)
    T_plane = np.eye(4)
    T_plane[:3, :3] = R_matrix
    T_plane[:3, 3] = t
    
    print("R: ", R_matrix)
    print("t: ", t)
    
    q = rotation_matrix_to_quaternion(R_matrix)
    q2 = R.from_matrix(R_matrix).as_quat()
    
    print("q: ", q)
    print("q2: ", q2)

    return T_plane

def plane_to_SE3_2(A, B, C, D):
    n = np.array([A, B, C])
    n = n / np.linalg.norm(n)
    
    # 生成一个不平行与n的向量
    if np.allclose(n, [0, 1, 0]):
        v = np.array([1, 0, 0])
    else:
        v = np.array([0, 1, 0])
        
    # 计算x轴和y轴
    x_axis = np.cross(v, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    # 组合旋转矩阵R
    Rmat = np.vstack((x_axis, y_axis, n)).T
    t = -D * n
    # q = rotation_matrix_to_quaternion(R)
    print(Rmat)
    q = R.from_matrix(Rmat).as_quat()
    print("q = ", q)
    print("t = ", t)
    
def plane_to_SE3(A, B, D):
    # 法向量
    normal = np.array([A, B, -1])
    normal_unit = normal / np.linalg.norm(normal)  # 归一化法向量

    # 平面上的一个点
    point_on_plane = np.array([0, 0, D])

    # 计算旋转矩阵
    z_axis = np.array([0, 0, 1]) # 不平行与法向量n的项链为z轴
    v = np.cross(normal_unit, z_axis)
    c = np.dot(normal_unit, z_axis)
    s = np.linalg.norm(v)

    if s == 0:
        # 平面与 z 轴平行的情况
        R = np.eye(3)
    else:
        v /= s
        # 计算旋转矩阵（使用 Rodrigues 公式）
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    # 平移向量
    t = point_on_plane
    # t = - D * normal_unit

    # 构建 SE(3)
    SE3 = np.eye(4)
    SE3[:3, :3] = R
    SE3[:3, 3] = t

    q = rotation_matrix_to_quaternion(R)
    
    print("q = ", q)
    print("t = ", t)
    return SE3

def estimate_plane():
    points = getData("/home/heda/zyy/dataset/phantom3_village-kfs/map.txt")

    # 使用 RANSAC 估计平面
    model = LinearRegression()
    ransac = RANSACRegressor(base_estimator=model, residual_threshold=1.0, random_state=0)
    ransac.fit(points[:, :2], points[:, 2])

    # 估计平面参数
    inlier_mask = ransac.inlier_mask_
    x_inliers = points[inlier_mask, 0]
    y_inliers = points[inlier_mask, 1]
    z_inliers = points[inlier_mask, 2]

    # 计算拟合的平面参数
    slope_x, slope_y = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_
    print(f"平面参数: z = {slope_x} * x + {slope_y} * y + {intercept}")
    
    # plane_to_SE3(slope_x, slope_y, intercept)
    # plane_to_SE3_2(slope_x, slope_y, intercept)

    # # P, Q, N = RANSAC(data)
    # # draw(points.T, slope_x, slope_y, intercept)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='所有点', alpha=0.5)
    # ax.scatter(x_inliers, y_inliers, z_inliers, color='red', label='内点', alpha=0.5)


    # print(x_inliers[0])
    # print(y_inliers[0])
    # print(z_inliers[0])
    # # 绘制估计的平面
    # xx, yy = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
    # zz = slope_x * xx + slope_y * yy + intercept
    # ax.plot_surface(xx, yy, zz, color='green', alpha=0.5)

    # ax.view_init(elev=10, azim=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.savefig("/home/heda/zyy/dataset/phantom3_village-kfs/plane_est.jpg")
    # # plt.show()
    return slope_x, slope_y, intercept

def transform_plane(a, b, c):
    # 定义相机坐标系下的平面法向量n和偏移量d
    n_w = np.array([a, b, 1]) 
    d_w = c
    
    Rgw = np.array([[189.21141, -32.361591, -15.425155],
                    [-32.733021, -189.74429, -3.4381607],
                    [-14.62044, 5.9999223, -191.92815]])
    tgw = np.array([-4.6523623, 259.50854, 165.72607])
    
    n_g = Rgw @ n_w
    d_g = d_w - np.dot(n_g, tgw)
    
    # 输出地图坐标系下的平面方程
    print("地图坐标系下的平面方程为: {}x + {}y + {}z + {} = 0".format(n_g[0], n_g[1], n_g[2], d_g))
    # slope_x = -n_g[0] / n_g[2]
    # slope_y = -n_g[1] / n_g[2]
    # intercept = -d_g / n_g[2]
    # plane_to_SE3(slope_x, slope_y, intercept)
    # plane_to_SE3_2(slope_x, slope_y, intercept)

def transform_plane2(a, b, d):
    # 定义相机坐标系下的平面法向量n和偏移量d
    n_w = np.array([a, b, 1]) 
    d_w = d
    
    Rgw = np.array([[188.5992, -32.144547, -25.942778],
                    [-32.767593, -190.2527, -2.4806659],
                    [-25.15123, 6.8262043, -191.30284]])
    # qgw = rotation_matrix_to_quaternion(Rgw) # w, x, y, z
    # print("四元数：", qgw)
    # rgw2 = quaternion_to_rotation_matrix(qgw)
    # print("旋转矩阵：", rgw2)
    # qgw2 = rotation_matrix_to_quaternion(rgw2)
    # print("四元数2：", qgw2)
    
    # rotation = R.from_matrix(Rgw)
    # quat = rotation.as_quat()
    # print("四元数-scipy：", quat)
    
    # rotation2 = R.from_quat(quat)
    # rotation_matrix = rotation2.as_matrix()
    # print("旋转矩阵-scipy:", rotation_matrix)
    
    # rotation3 = R.from_matrix(rotation_matrix)
    # quat2 = rotation3.as_quat()
    # print("四元数-scipy2：", quat2)
    tgw = np.array([-4.6330438, 259.57529, 165.72015])
    
    n_g = Rgw @ n_w
    d_g = Rgw @ np.array([0,0,d]) + tgw
    print(n_g)
    print(d_g)
    d_g = np.dot(n_g, d_g) # 平面法向量和平面上一点的点积
    print(d_g)
    # 输出地图坐标系下的平面方程
    print("地图坐标系下的平面方程为: {}x + {}y + {}z = {}".format(n_g[0]/n_g[2], n_g[1]/n_g[2], n_g[2]/n_g[2], d_g/n_g[2]))
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='所有点', alpha=0.5)
    # # ax.scatter(x_inliers, y_inliers, z_inliers, color='red', label='内点', alpha=0.5)
    
    # xx, yy = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
    # zz = -n_g[0]/n_g[2] * xx - n_g[1]/n_g[2] * yy + d_g/n_g[2]
    # ax.plot_surface(xx, yy, zz, color='green', alpha=0.5)

    # ax.view_init(elev=10, azim=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.savefig("/home/heda/zyy/dataset/phantom3_village-kfs/plane_est_from_transform.jpg")
    return n_g[0]/n_g[2], n_g[1]/n_g[2], n_g[2]/n_g[2], d_g/n_g[2]

def verify_quaternion():
    q_pose = np.array([0.999997, -0.00108932, -0.00187957, 0.00144478]) # w, x, y, z
    r_pose = quaternion_to_rotation_matrix(q_pose)
    q_cyc = rotation_matrix_to_quaternion(r_pose)
    t_pose = np.array([0.00752067, 0.11656, 0.0037729])
    print("q", q_pose)
    print("r", r_pose)
    print("q_cyc", q_cyc)
    
    q_plane = np.array([0.006, 0.0, 0.0, 0.99998]) # w, x, y, z
    r_plane = quaternion_to_rotation_matrix(q_plane)
    q_cyc = rotation_matrix_to_quaternion(r_plane)
    print("q", q_plane)
    print("r", r_plane)
    print("q_cyc", q_cyc)

    t_plane = r_plane.T @ t_pose.T
    print("t_plane: ", t_plane)
    
def transform_points():
    # path = "/home/heda/zyy/dataset/phantom3_village-kfs/trajectory_self.txt"
    # map_points = []
    # with open(path, 'r') as f:
    #     lines = f.readlines()
    #     lines = [line.strip() for line in lines]
    #     for line in lines[0:]:
    #         cleaned_data = ''.join(filter(lambda x: x.isprintable(), line))
    #         _, x, y, z, _, _, _, _ = cleaned_data.split()
    #         x = float(x)
    #         y = float(y)
    #         z = float(z)
    #         map_points.append([x, y, z])
    points = getData("/home/heda/zyy/dataset/phantom3_village-kfs/map.txt")
    Rgw = np.array([[188.5992, -32.144547, -25.942778],
                    [-32.767593, -190.2527, -2.4806659],
                    [-25.15123, 6.8262043, -191.30284]])
    tgw = np.array([-4.6330438, 259.57529, 165.72015])
    
    transform_points = []
    for point in points:
        transform_point = Rgw @ point + tgw
        # print(transform_point)
        transform_points.append(transform_point)
    
    transform_points = np.array(transform_points)
    model = LinearRegression()
    ransac = RANSACRegressor(base_estimator=model, residual_threshold=1.0, random_state=0)
    ransac.fit(transform_points[:, :2], transform_points[:, 2])

    # 估计平面参数
    inlier_mask = ransac.inlier_mask_
    x_inliers = transform_points[inlier_mask, 0]
    y_inliers = transform_points[inlier_mask, 1]
    z_inliers = transform_points[inlier_mask, 2]

    # 计算拟合的平面参数
    slope_x, slope_y = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_
    print(f"平面参数: z = {slope_x} * x + {slope_y} * y + {intercept}")
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='所有点', alpha=0.5)
    # # ax.scatter(x_inliers, y_inliers, z_inliers, color='red', label='内点', alpha=0.5)
    
    # xx, yy = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
    # zz = slope_x * xx + slope_y * yy + intercept
    # ax.plot_surface(xx, yy, zz, color='green', alpha=0.5)

    # ax.view_init(elev=10, azim=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.savefig("/home/heda/zyy/dataset/phantom3_village-kfs/plane_est_from_points.jpg")
    return transform_points

def transform_plane():
    p1 = np.array([0, 0, 1])
    p2 = np.array([0, 1, 1])
    p3 = np.array([1, 0, 1])
    
    Rgw = np.array([[188.5992, -32.144547, -25.942778],
                    [-32.767593, -190.2527, -2.4806659],
                    [-25.15123, 6.8262043, -191.30284]])
    tgw = np.array([-4.6330438, 259.57529, 165.72015])
    
    
    transform_p1 = Rgw @ p1 + tgw
    transform_p2 = Rgw @ p2 + tgw
    transform_p3 = Rgw @ p3 + tgw
    
    print("p1: ", transform_p1)
    print("p2: ", transform_p2)
    print("p3: ", transform_p3)
    
    line1 = transform_p1 - transform_p2
    line2 = transform_p2 - transform_p3
    normal = np.cross(line1, line2)

    # 提取法向量的分量 a, b, c
    a, b, c = normal

    # 计算平面方程的常数 d
    d = np.dot(normal, p1)  # 使用点 P1 来计算 d
    print("地图坐标系下的平面方程为: {}x + {}y + {}z = {}".format(a/c, b/c, c/c, d/c))
    return a, b, c, d
    
if __name__ == '__main__':
    # transform_plane()
    # plane_to_SE3_2(0, 0, 1)
    # points = transform_points()
    # estimate_plane()
    # # verify_quaternion()
    # # a, b, d = estimate_plane()
    # # 0. 平面z=1 
    # nc = np.array([0,0,-1])
    # dc = 1
    # plane_to_se3_3(nc,dc)
    
    # 1. z=1先转到gps坐标系的平面公式，再按照平面公式转成se3
    a = 0
    b = 0
    d = 1
    # a = -0.0067525795829590125
    # b = 0.008150534046932365
    # d = 1.0095331519137933
    # plane_to_se3_3(np.array([a, b, -1]), d)
    nx, ny, nz, nd = transform_plane2(a, b, d)
    n = np.array([nx, ny, nz])
    plane_to_se3_3(n, nd)
    # plane_to_SE3_2(nx, ny, nz, nd)
    
    # 2. z=1先按照平面公式转成se3,再通过坐标变换到gps坐标系
    # plane_t = np.array([0,0,1])
    # plane_r = np.array([[1,0,0],
    #                     [0,1,0],
    #                     [0,0,1]])
    # Rgw = np.array([[188.5992, -32.144547, -25.942778],
    #                 [-32.767593, -190.2527, -2.4806659],
    #                 [-25.15123, 6.8262043, -191.30284]])
    # tgw = np.array([-4.6330438, 259.57529, 165.72015])
    # Rgpsc = Rgw @ plane_r
    # tgps = tgw @ plane_t + tgw
    # print(Rgpsc)
    # print(tgps)
    