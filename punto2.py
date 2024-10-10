import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

def y(x):
    return x**5 - 8*x**3 + 10*x + 6

def grad_y(x):
    return derivative(y, x, dx=1e-6)

def hessian_y(x):
    return derivative(grad_y, x, dx=1e-6)

def newton_raphson(grad_y, hessian_y, x0, alpha=0.6, tolerance=0.001):
    x = x0
    iteration = 0
    while abs(grad_y(x)) > tolerance:
        x_next = x - alpha * grad_y(x) / hessian_y(x)
        if abs(x_next - x) < tolerance:
            break
        x = x_next
        iteration += 1
    return x

def encontrar_puntos_criticos(grad_y, hessian_y, intervalo, paso=0.5, alpha=0.6, tolerance=0.001):
    puntos_criticos = []
    for x0 in np.arange(intervalo[0], intervalo[1], paso):
        punto_critico = newton_raphson(grad_y, hessian_y, x0, alpha, tolerance)
        if not any(np.isclose(punto_critico, p[0], atol=1e-3) for p in puntos_criticos):  # Evitar duplicados
            tipo = "Máximo" if hessian_y(punto_critico) < 0 else "Mínimo" if hessian_y(punto_critico) > 0 else "Indefinido"
            puntos_criticos.append((punto_critico, y(punto_critico), tipo))
    return puntos_criticos


def encontrar_extremos_globales(puntos_criticos):
    max_global = max(puntos_criticos, key=lambda p: p[1])
    min_global = min(puntos_criticos, key=lambda p: p[1])
    return max_global, min_global


intervalo = (-3, 3)

puntos_criticos = encontrar_puntos_criticos(grad_y, hessian_y, intervalo)
max_global, min_global = encontrar_extremos_globales(puntos_criticos)

# Mostramos los puntos críticos
print("Puntos críticos encontrados:")
for punto in puntos_criticos:
    print(f"x = {punto[0]:.4f}, y = {punto[1]:.4f}, Tipo: {punto[2]}")

print(f"\nMáximo global: x = {max_global[0]:.4f}, y = {max_global[1]:.4f}")
print(f"Mínimo global: x = {min_global[0]:.4f}, y = {min_global[1]:.4f}")

# Graficamos la función y los puntos críticos
x_vals = np.linspace(intervalo[0], intervalo[1], 400)
y_vals = y(x_vals)

plt.figure(figsize=(20, 10))
plt.plot(x_vals, y_vals, label="y(x)", color='blue')

for punto in puntos_criticos:
    plt.plot(punto[0], punto[1], 'ko', label=f"{punto[2]} en x={punto[0]:.2f}")

plt.plot(max_global[0], max_global[1], 'ro', label=f"Máximo global en x={max_global[0]:.2f}")
plt.plot(min_global[0], min_global[1], 'ro', label=f"Mínimo global en x={min_global[0]:.2f}")

plt.legend()
plt.title("Máximos y mínimos locales y globales")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.savefig('punto2.png')
