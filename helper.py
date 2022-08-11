from vectors import Vector, Matrix
from pygame import draw


BLACK = (0., 0., 0.)
WHITE = (255, 255, 255)


def to_2d_point(vec: Vector):
    return int(vec[0]), int(vec[1])


def clamp(val, minn, maxn):
    if val < minn:
        return minn
    if val > maxn:
        return maxn
    return val


def nmap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def fill_bottom_flat_triangle(v1, v2, v3, surface, color=WHITE):
    try:
        slope1 = (v2[0] - v1[0]) // (v2[1] - v1[1])
        slope2 = (v3[0] - v1[0]) // (v3[1] - v1[1])
    except ZeroDivisionError:
        print(v1, v2, v3)
        return

    curx1 = v1[0]
    curx2 = v1[0]

    for scanline_y in range(v1[1], v2[1] + 1):
        draw.line(surface, color, (curx1, scanline_y), (curx2, scanline_y))
        curx1 += slope1
        curx2 += slope2


def fill_top_flat_triangle(v1, v2, v3, surface, color=WHITE):
    try:
        slope1 = (v3[0] - v1[0]) // (v3[1] - v1[1])
        slope2 = (v3[0] - v2[0]) // (v3[1] - v2[1])
    except ZeroDivisionError:
        print(v1, v2, v3)
        return

    curx1 = v3[0]
    curx2 = v3[0]

    for scanline_y in range(v3[1], v1[1] - 1, -1):
        draw.line(surface, color, (curx1, scanline_y), (curx2, scanline_y))
        curx1 -= slope1
        curx2 -= slope2


def fill_triangle_2d(pts, surface, color=WHITE):
    pts.sort(key=lambda x: x[1])
    p1, p2, p3 = pts

    if p2[1] == p3[1]:
        fill_bottom_flat_triangle(p1, p2, p3, surface, color)
    elif p1[1] == p2[1]:
        fill_top_flat_triangle(p1, p2, p3, surface, color)
    else:
        slope = (p3[0] - p1[0]) / (p3[1] - p1[1])
        dy = float(p2[1] - p1[1])
        dx = int(dy * slope)
        p4 = (p1[0] + dx, p2[1])
        fill_bottom_flat_triangle(p1, p2, p4, surface, color)
        fill_top_flat_triangle(p2, p4, p3, surface, color)


def vector3_by_matrix4(v, m):
    if len(v) != len(m):
        v.append(1.)
    v *= m
    if v[3]:
        v /= v[3]
    return Vector(v[:3])
