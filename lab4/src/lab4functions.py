# coding=utf-8
import numpy as np
from copy import copy

cos = np.cos
sin = np.sin
pi = np.pi


# Obtención de la Matriz Homogenea (T) de DH:
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

# Obtención de la Matriz Homogenea del origen al efector final (posición del efector final): (0-T-6)


def fkine_ur5(q):
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

# obtención del Jacobiano (se recibe el q actual y la variación que se lehará)


def jacobian_ur5(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma
    como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]
    """
    # Crear una matriz 3x6
    J = np.zeros((3, 6))
    # Transformacion homogenea (0-T-6) usando el q dado
    T = fkine_ur5(q)
    # Iteracion para la derivada de cada columna
    for i in xrange(6):
        # Copiar la configuracion articular(q) y almacenarla en dq
        dq = copy(q)
        # Incrementar los valores de cada q sumandoles un delta a cada uno
        dq[i] = dq[i] + delta
        # Obtención de la nueva Matriz Homogenea con los nuevos valores articulares, luego del incremento(q+delta)
        Td = fkine_ur5(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        for j in xrange(3):
            J[j][i] = (Td[j][3]-T[j][3])/delta
    return J


# Método de Newton, se recibe la posición a la que se desea ir(xdes) y elvalor actual del q
# responde a: ¿cuáles son los valores articulares(q) que debe tener el robot para estar en la posición deseada?
def ikine_ur5(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la
    configuracion articular inicial de q0.
    Emplear el metodo de newton
    """
    epsilon = 0.001  # error(distancia) mínima requerida
    max_iter = 1000  # max cantidad de iteraciones
    delta = 0.00001

    # se copia el valor de la articulación inicial(q0) y se almacena en q
    q = copy(q0)
    for i in range(max_iter):
        J = jacobian_ur5(q, delta)  # Matriz Jacobiana
        Td = fkine_ur5(q)  # Matriz Actual
        # Posicion Actual: extrayendo la parte traslacional de la Matriz Homogenea(d1,d2,d3) y almacenandolos en el vector xact
        xact = Td[0:3, 3]
        # Error entre pos deseada y pos actual (cuanta distancia separa ambos puntos)
        e = xdes-xact  # difencia de ambos, dado que son 2 vectores con mismo origen, la resta significa un vector que va desde xact a xdes
        # e=var(x,y,z) -> variación espacial para llegar a pos deseada
        # Metodo de Newton (se actualiza el valor de q y vuelve al loop si tdv no llega a la max iteración)
        # q=q+var(q) -> q=q+(inv(J)*var(x, y, z)) -> q=q+(inv(J)*e)
        q = q+np.dot(np.linalg.pinv(J), e)
        # Condicion de termino
        # norma=modulo, el modulo es ladistancia en magnitud faltante para llegar a la posición deseada
        if(np.linalg.norm(e) < epsilon):
            break
        pass
    return q

# Lo mismo que se realizó anteriormente, solo cambiando la línea de método de Newton por el del Metodo de la Gradiente
# Este método necesita además del parámetro delta


def ik_gradient_ur5(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la
    configuracion articular inicial de q0.
    Emplear el metodo gradiente
    """
    epsilon = 0.001
    max_iter = 1000
    delta = 0.00001
    alpha = 0.5
    q = copy(q0)
    for i in range(max_iter):
        # Main loop
        # Matriz Jacobiana
        J = jacobian_ur5(q, delta)
        # Matriz Actual
        Td = fkine_ur5(q)
        # Posicion Actual
        xact = Td[0:3, 3]
        # Error entre pos deseada y pos actual
        e = xdes-xact
        # Metodo de la gradiente
        q = q+alpha*np.dot(J.T, e)
        # Condicion de termino
        if(np.linalg.norm(e) < epsilon):
            break
        pass
    return q
