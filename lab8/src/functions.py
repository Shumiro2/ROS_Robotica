# coding=utf-8

import numpy as np
from copy import copy
pi = np.pi


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine_sawyer(q):
    T0 = dh(0.080, 0, 0, 0)
    T1 = dh(0.237, q[0], 0.081, -pi/2)
    T2 = dh(0.1925, q[1]-pi/2, 0, -pi/2)
    T3 = dh(0.400, q[2], 0, pi/2)
    T4 = dh(-0.1685, q[3], 0, -pi/2)
    T5 = dh(0.400, q[4], 0, pi/2)
    T6 = dh(0.1363, q[5], 0, -pi/2)
    T7 = dh(0.13375, q[6], 0, 0)
    T = T0.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5).dot(T6).dot(T7)
    return T


def jacobian_sawyer(q,delta=0.0001):
    J=np.zeros((3,8))

    T=fkine_sawyer(q)

    for i in xrange(8):
        dq=copy(q)
        dq[i]=dq[i]+delta
        Td=fkine_sawyer(dq)
        for j in xrange(3):
            J[j][i] = (Td[j][3]-T[j][3])/delta        
    return J

def ikine_sawyer(xdes, q0):
    epsilon = 0.001
    max_iter = 1000
    delta = 0.00001
    q = copy(q0)
    for i in range(max_iter):
        J = jacobian_sawyer(q, delta)
        T = fkine_sawyer(q)
        f = T[0:3, 3]
        e = xdes - f
        q = q + np.dot(J.T, e)
        if(np.linalg.norm(e) < epsilon):
            break
    return q



