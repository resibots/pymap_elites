import numpy as np
from math import cos, sin, pi, sqrt

def circle_segment_intersection(circle_center, circle_radius, pt1, pt2, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.
    From: https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
        intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

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

    # this requires the fw kinematics to be computed first
    def collides(self, circle_center, circle_radius):
        intersections = []
        for i in range(0, len(self.joint_xy) - 1):
            intersections += circle_segment_intersection(circle_center, circle_radius, self.joint_xy[i], self.joint_xy[i + 1])
        return intersections            

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