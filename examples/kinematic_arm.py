import numpy as np
from math import cos, sin, pi, sqrt

class Arm:
    def __init__(self, lengths):
        self.n_dofs = len(lengths)
        self.lengths = np.concatenate(([0], lengths))
        self.joint_xy = []

    def fw_kinematics(self, p):
        assert(len(p) == self.n_dofs)
        p = np.append(p, 0)
        self.joint_xy = []
        mat = np.matrix(np.identity(4))
        for i in range(0, self.n_dofs + 1):
            m = [[cos(p[i]), -sin(p[i]), 0, self.lengths[i]],
                 [sin(p[i]),  cos(p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            self.joint_xy += [v[0:2].A.flatten()]
        return self.joint_xy[self.n_dofs], self.joint_xy

if __name__ == "__main__":
    # 1-DOFs
    a = Arm([1])
    v,_ = a.fw_kinematics([0])
    np.testing.assert_almost_equal(v, [1, 0])
    v,_ = a.fw_kinematics([pi/2])
    np.testing.assert_almost_equal(v, [0, 1])

    # 2-DOFs
    a = Arm([1, 1])
    v,_ = a.fw_kinematics([0, 0])
    np.testing.assert_almost_equal(v, [2, 0])
    v,_ = a.fw_kinematics([pi/2, 0])
    np.testing.assert_almost_equal(v, [0, 2])
    v,_ = a.fw_kinematics([pi/2, pi/2])
    np.testing.assert_almost_equal(v, [-1, 1])
    v,x = a.fw_kinematics([pi/4, -pi/2])
    np.testing.assert_almost_equal(v, [sqrt(2), 0])

    # a 4-DOF square
    a = Arm([1, 1, 1,1])
    v,_ = a.fw_kinematics([pi/2, pi/2, pi/2, pi/2])
    np.testing.assert_almost_equal(v, [0, 0])