import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


x_sym, y_sym = sp.symbols('x y')

z_sym = (x_sym - 1)**2 + 100 * (y_sym - x_sym**2)**2

grad_z_sym = [sp.diff(z_sym, var) for var in (x_sym, y_sym)]  # Gradiente
hessian_z_sym = sp.hessian(z_sym, (x_sym, y_sym))  # Hessiana


grad_z = sp.lambdify((x_sym, y_sym), grad_z_sym, 'numpy')
hessian_z = sp.lambdify((x_sym, y_sym), hessian_z_sym, 'numpy')

z = sp.lambdify((x_sym, y_sym), z_sym, 'numpy')

def newton_raphson(grad_z, hessian_z, x0, y0, alpha=1, tolerance=0.001, max_iter=100):
    x, y = x0, y0
    iteraciones = [(x, y)]  # Para almacenar los puntos de cada iteración

    grad = np.array(grad_z(x, y))
    while np.linalg.norm(grad) > tolerance:
        grad = np.array(grad_z(x, y))
        hess = np.array(hessian_z(x, y))
        
        delta = -alpha * np.linalg.inv(hess) @ grad
        x, y = x + delta[0], y + delta[1]
        iteraciones.append((x, y))  # Guardamos la iteración
    
    return np.array([x, y]), iteraciones

# Parámetros iniciales
x0, y0 = 0, 10 
alpha = 1 
tolerance = 0.001


sol, iteraciones = newton_raphson(grad_z, hessian_z, x0, y0, alpha=alpha, tolerance=tolerance)


x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-5, 11, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = z(X, Y)

# Mostramos el mínimo encontrado
print(f"El mínimo encontrado es en x = {sol[0]:.4f}, y = {sol[1]:.4f}. z={z(sol[0],sol[1])}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)


x_iters, y_iters = zip(*iteraciones)  # Coordenadas x y de las iteraciones
z_iters = [z(x_i, y_i) for x_i, y_i in iteraciones]  # Coordenadas z correspondientes


ax.plot(x_iters, y_iters, z_iters, color='cyan', linewidth=2, label='Ruta de iteraciones')

x_min, y_min = sol
z_min = z(x_min, y_min)
ax.scatter(x_min, y_min, z_min, color='red', s=100, label='Mínimo')

ax.set_title('Superficie de la función con la ruta de puntos encontrados (cyan) y el mínimo (rojo)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Mostrar leyenda y gráfica
plt.legend()
plt.savefig('punto3.png')
