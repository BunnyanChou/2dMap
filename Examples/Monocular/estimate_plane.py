import random
import numpy as np
from math import acos, sin, cos, fabs, sqrt, log
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

def new_csv():
    with open("pcltest.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[1, 2, 3, 4, ',']])

def getData(filepath, row_need=1000):
    map_points = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            cleaned_data = ''.join(filter(lambda x: x.isprintable(), line))
            _, x, y, z = cleaned_data.split()
            x = float(x)
            y = float(y)
            z = float(z)   
            map_points.append([x, y, z])
    return np.array(map_points)

def solve_plane(A, B, C):
    # 两个常量
    N = np.array([0, 0, 1])
    Pi = 3.1415926535
    
    # 计算平面的单位法向量，即BC与BA的叉积
    Nx = np.cross(B - C, B - A)
    Nx = Nx / np.linalg.norm(Nx)
    
    # 计算单位选择向量与旋转角（范围0到pi）
    Nv = np.cross(Nx, N)
    angle = acos(np.dot(Nx, N))
    
    # 考虑到两个向量夹角不大于Pi / 2, 这里需要处理一下
    if angle > Pi / 2.0:
        angle = Pi - angle
        Nv = -Nv
    
    # 确定平面上一个点
    Point = B
    # 计算四元数
    Quaternion = np.append(Nv * sin(angle / 2), cos(angle / 2))
    
    return Point, Quaternion, Nx

def solve_distance(M, P, N):
    """
    求解点M到平面（P， Q）的距离
    """
    A = N[0]
    B = N[1]
    C = N[2]
    D = -A * P[0] - B * P[1] - C * P[2]
    
    return fabs(A * M[0] + B * M[1] + C * M[2] + D) / sqrt(A ** 2 + B ** 2 + C ** 2)

def RANSAC(data):
    SIZE = data.shape[0]
    
    iters = 10000
    sigma = 0.15
    pretotal = 0
    per = 0.999
    P = np.array([])
    Q = np.array([])
    N = np.array([])
    for i in range(iters):
        sample_index = random.sample(range(SIZE), 3)
        P, Q, N = solve_plane(data[sample_index[0]], data[sample_index[1]], data[sample_index[2]])
        
        # 算出内点数目
        total_inlier = 0
        for index in range(SIZE):
            if solve_distance(data[index], P, N) < sigma:
                total_inlier = total_inlier + 1
                
        if  total_inlier > pretotal:
            # print(total_inlier / SIZE)
            iters = log(1 - per) / log(1 - pow(abs(total_inlier / SIZE), 2))
            pretotal = total_inlier
        
        if total_inlier > SIZE / 2:
            break
    return P, Q, N

def draw(data, A_plane, B_plane, D_plane):
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.scatter(data[0], data[1], data[2], c="gold")
    
    x = np.linspace(data[0].min(), data[0].max(), 10)
    y = np.linspace(data[1].min(), data[1].max(), 10)
    X, Y = np.meshgrid(x, y)
    # Z = -(N[0] * X + N[1] * Y - (N[0] * P[0] + N[1] * P[1] + N[2] * P[2])) / N[2]
    Z = A_plane * X + B_plane * Y + D_plane
    # ax.plot_surface(X, Y, Z)
    
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    
    # plt.show()
    plt.savefig("/home/heda/zyy/dataset/phantom3_village-kfs/plane.jpg")
    
def test():
    A = np.random.randn(3)
    B = np.random.randn(3)
    C = np.random.randn(3)
    
    P, Q, N = solve_plane(A, B, C)
    
    D = np.random.randn(3)
    d = solve_distance(D, P, N)

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

def plane_to_SE3(A, B, D):
    # 法向量
    normal = np.array([A, B, -1])
    normal_unit = normal / np.linalg.norm(normal)  # 归一化法向量

    # 平面上的一个点
    point_on_plane = np.array([0, 0, D])

    # 计算旋转矩阵
    z_axis = np.array([0, 0, 1])
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
    
    plane_to_SE3(slope_x, slope_y, intercept)

    # P, Q, N = RANSAC(data)
    # draw(points.T, slope_x, slope_y, intercept)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='所有点', alpha=0.5)
    ax.scatter(x_inliers, y_inliers, z_inliers, color='red', label='内点', alpha=0.5)


    print(x_inliers[0])
    print(y_inliers[0])
    print(z_inliers[0])
    # 绘制估计的平面
    xx, yy = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
    zz = slope_x * xx + slope_y * yy + intercept
    ax.plot_surface(xx, yy, zz, color='green', alpha=0.5)

    ax.view_init(elev=10, azim=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig("/home/heda/zyy/dataset/phantom3_village-kfs/plane_est.jpg")
    # plt.show()
    
if __name__ == '__main__':
    q_pose = np.array([0.999997, -0.00108932, -0.00187957, 0.00144478])
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
    
    
    