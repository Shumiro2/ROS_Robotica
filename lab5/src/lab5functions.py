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


def fkine(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]

    """
    # Longitudes (en metros)
    l1 = 0.0892
    l2 = 0.425
    l3 = 0.392
    l4 = 0.1093
    l5 = 0.09475
    l6 = 0.0825

    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T1 = dh(l1, q[0], 0, pi/2)
    T2 = dh(l4, q[1]+pi, l2, 0)
    T3 = dh(-l4, q[2], l3, 0)
    T4 = dh(l4, q[3]+pi, 0, pi/2)
    T5 = dh(l5, q[4]+pi, 0, pi/2)
    T6 = dh(l6, q[5], 0, 0)
    # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)

    return T


def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    # Crear una matriz 3x6
    J = np.zeros((3, 6))
    # Transformacion homogenea (0-T-6) usando el q dado
    T = fkine(q)
    # Iteracion para la derivada de cada columna
    for i in xrange(6):
        # Copiar la configuracion articular(q) y almacenarla en dq
        dq = copy(q)
        # Incrementar los valores de cada q sumandoles un delta a cada uno
        dq[i] = dq[i] + delta
        # Obtención de la nueva Matriz Homogenea con los nuevos valores articulares, luego del incremento(q+delta)
        Td = fkine(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        for j in xrange(3):
            J[j][i] = (Td[j][3]-T[j][3])/delta
    return J


def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """

    # Alocacion de memoria
    J = np.zeros((7, 6))
    J1 = np.zeros((3, 6))
    J2 = np.zeros((4, 6))
    # Transformacion homogenea inicial (usando q)
    T = fkine(q)
    R = T[0:3, 0:3]
    Q = rot2quat(R)
    # Iteracion para la derivada de cada columna
    for i in xrange(6):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq1 = copy(q)
        # Incrementar los valores de cada q sumandoles un delta a cada uno
        dq1[i] = dq1[i] + delta
        # Obtención de la nueva Matriz Homogenea con los nuevos valores articulares, luego del incremento(q+delta)
        Td = fkine(dq1)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        for j in xrange(3):
            J1[j][i] = (Td[j][3]-T[j][3])/delta

    for i in xrange(6):
        dq2 = copy(q)
        dq2[i] = dq2[i] + delta
        Td2 = fkine(dq2)
        Rd = Td2[0:3, 0:3]
        Qd = rot2quat(Rd)
        for j in xrange(4):
            J2[j][i] = (Qd[j] - Q[j])/delta

    J = np.concatenate((J1, J2))
    # Implementar este Jacobiano aqui

    return J


def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0., ]

    quat[0] = 0.5*np.sqrt(R[0, 0]+R[1, 1]+R[2, 2]+1.0)
    if (np.fabs(R[0, 0]-R[1, 1]-R[2, 2]+1.0) < dEpsilon):
        quat[1] = 0.0
    else:
        quat[1] = 0.5*np.sign(R[2, 1]-R[1, 2]) * \
            np.sqrt(R[0, 0]-R[1, 1]-R[2, 2]+1.0)
    if (np.fabs(R[1, 1]-R[2, 2]-R[0, 0]+1.0) < dEpsilon):
        quat[2] = 0.0
    else:
        quat[2] = 0.5*np.sign(R[0, 2]-R[2, 0]) * \
            np.sqrt(R[1, 1]-R[2, 2]-R[0, 0]+1.0)
    if (np.fabs(R[2, 2]-R[0, 0]-R[1, 1]+1.0) < dEpsilon):
        quat[3] = 0.0
    else:
        quat[3] = 0.5*np.sign(R[1, 0]-R[0, 1]) * \
            np.sqrt(R[2, 2]-R[0, 0]-R[1, 1]+1.0)

    return np.array(quat)


def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = rot2quat(T[0:3, 0:3])
    res = [T[0, 3], T[1, 3], T[2, 3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)


def skew(w):
    R = np.zeros([3, 3])
    R[0, 1] = -w[2]
    R[0, 2] = w[1]
    R[1, 0] = w[2]
    R[1, 2] = -w[0]
    R[2, 0] = -w[1]
    R[2, 1] = w[0]
    return R
