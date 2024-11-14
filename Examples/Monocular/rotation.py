import numpy as np
from scipy.spatial.transform import Rotation as R

# # 第一个矩阵
# R1 = np.array([[0.96906198, -0.22236039, -0.10712017],
#                [-0.22577405, -0.97395942, -0.02071553],
#                [-0.09972438, 0.04425959, -0.99403025]])

# # 第二个矩阵
# R2 = np.array([[189.21141, -32.361591, -15.425155],
#                [-32.733021, -189.74429, -3.4381607],
#                [-14.62044, 5.9999223, -191.92815]])

# scale = R2[0,0] / R1[0, 0]
# print(scale)
# R3 = R1 * scale
# print(R3)
# # 检查矩阵是否正交 (R^T * R == I)
# def is_orthogonal(matrix):
#     identity_matrix = np.eye(3)
#     return np.allclose(np.dot(matrix.T, matrix), identity_matrix)

# # 检查矩阵的行列式是否为 1
# def is_unit_determinant(matrix):
#     return np.isclose(np.linalg.det(matrix), 1)

# # 验证第一个矩阵
# print("第一个矩阵是正交矩阵:", is_orthogonal(R1))
# print("第一个矩阵的行列式为 1:", is_unit_determinant(R1))

# # 验证第二个矩阵
# print("第二个矩阵是正交矩阵:", is_orthogonal(R2))
# print("第二个矩阵的行列式为 1:", is_unit_determinant(R2))

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

Rwc = np.array([[188.20575, -33.326015, -27.274973],
              [-34.109985, -190.00569, -3.2103992],
              [-26.287951, 7.9482365, -191.10658]])
# Rwc_norm = Rwc / 193.04
qwc = R.from_matrix(Rwc).as_quat()
# qwc_norm = R.from_matrix(Rwc_norm).as_quat()
# print("Rwc_norm: ", Rwc_norm)
print("qwc: ", qwc)
# print("qwc_norm: ", qwc_norm)
q2 = rotation_matrix_to_quaternion(Rwc)
print("q2: ", q2)
