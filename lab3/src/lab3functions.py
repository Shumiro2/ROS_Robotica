
import numpy as np
from copy import copy

cos = np.cos
sin = np.sin
pi = np.pi


def dh(d, theta, a, alpha):

    cth = cos(theta)
    sth = sin(theta)
    ca = cos(alpha)
    sa = sin(alpha)
    T = np.array([[cth, -ca*sth, sa*sth, a*cth],
                  [sth, ca*cth, -sa*cth, a*sth],
                  [0, sa, ca, d],
                  [0, 0, 0, 1]])
    return T


def fkine_ur5(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores
    articulares.
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    """
    # Longitudes (en metros)
    l1 = 0.0892
    l2 = 0.425
    l3 = 0.392
    l4 = 0.1093
    l5 = 0.09475
    l6 = 0.0825
    # Matrices DH (completar), emplear la funcion dh con los
    # parametros DH para cada articulacion
    T1 = dh(l1, q[0], 0, pi/2)
    T2 = dh(l4, q[1]+pi, l2, 0)
    T3 = dh(-l4, q[2], l3, 0)
    T4 = dh(l4, q[3]+pi, 0, pi/2)
    T5 = dh(l5, q[4]+pi, 0, pi/2)
    T6 = dh(l6, q[5], 0, 0)
    # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
    return T
