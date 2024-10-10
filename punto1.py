import scipy as sc
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.misc import derivative

def y(x):
    return 3*x**3 - 10*x**2 -56*x +50

x_values = np.linspace(-10, 10, 100)
y_values = y(x_values)

plt.figure()
plt.plot(x_values, y_values)
plt.savefig('punto1-graph.png')

def grad_y(x):
    return derivative(y, x, dx=1e-6)

def hessian_y(x):
    return derivative(grad_y, x, dx=1e-6)

def newton_raphson(grad_y, hessian_y, x0, alpha=0.6, tolerance=0.001):
    iteraciones = {}  # Diccionario para almacenar los puntos de cada suposición inicial
    for i in x0:
        x = i
        puntos_iteracion = [x]  # Almacenamos el punto inicial
        iteration = 0
        
        while abs(grad_y(x)) > tolerance:
            x_next = x - alpha * grad_y(x) / hessian_y(x)
            x = x_next
            iteration += 1
            puntos_iteracion.append(x)  
            print(f"Iteración {iteration}: x = {x}, gradiente = {grad_y(x)}")
        
        iteraciones[i] = puntos_iteracion  # Guardamos los puntos de la iteración para este x0
    return iteraciones

# Parámetros iniciales
x0 = [-4, 6]  # Se pusieron dos para hallar ambos puntos críticos

# ALPHA = 0.6
alpha = 0.6  
iteraciones = newton_raphson(grad_y, hessian_y, x0, alpha=alpha)

x_vals = np.linspace(-6, 6, 400)
y_vals = y(x_vals)

plt.figure()
plt.plot(x_vals, y_vals, label="y(x)")


for x_init, puntos in iteraciones.items():
    for j, punto in enumerate(puntos):
        if j < len(puntos) - 1:  
            plt.plot(punto, y(punto), 'ro', label=f"Iteración de x0={x_init:.2f}, α={alpha}" if j == 0 else "")
        else: 
            plt.plot(punto, y(punto), 'bo', label=f"Punto crítico de x0={x_init:.2f}, α={alpha}")

plt.legend()
plt.title(f"Método de Newton-Raphson con iteraciones y puntos críticos (α={alpha})")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.savefig('punto1-alpha0,6.png')


# ALPHA = 1
alpha = 1
iteraciones = newton_raphson(grad_y, hessian_y, x0, alpha=alpha)

x_vals = np.linspace(-6, 6, 400)
y_vals = y(x_vals)


plt.plot(x_vals, y_vals, label="y(x)")


for x_init, puntos in iteraciones.items():
    for j, punto in enumerate(puntos):
        if j < len(puntos) - 1: 
            plt.plot(punto, y(punto), 'go', label=f"Iteración de x0={x_init:.2f}, α={alpha}" if j == 0 else "")
        else: 
            plt.plot(punto, y(punto), 'bo', label=f"Punto crítico de x0={x_init:.2f}, α={alpha}")

plt.legend()
plt.title(f"Método de Newton-Raphson con iteraciones y PC (α={alpha}) and (α=0,6)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.savefig('punto1-doubleAlpha.png')