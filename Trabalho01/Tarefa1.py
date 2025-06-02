import numpy as np
import matplotlib.pyplot as plt

def horner(x0, a):
    
    '''
        O Método de Horner é um algoritmo para calcular o polinômio p(x0) e p'(x0)
        
        x0 - O valor que será avaliado
        a - Um array de coeficientes
        
        O grau do polinômio é "setado" igual ao número de coeficientes
    '''
    
    n = len(a) -1
    
    y = a[n]
    z = a[n]
    
    for j in range(n-1, 0, -1):
        y = x0 * y + a[j]
        z = x0 * z + y
    
    y = x0 * y + a[0]
    
    return y, z

def newton_horner(x0, a):
    
    '''
        O Método de Newton-Horner é usado para encontrar a raiz de um polinômio
        usando o Método de Newton com o polinômio e sua derivada calculada pelo 
        Método de Horner.
    '''
    itmax = 100 #quantidade máximo de iterações
    tol = 0.00000000000000001 #tolerância máxima de erro
    it = 0 #quantidade atual de iterações
    
    
    y, z = horner(x0, a)
    
    while ( abs(eval_p(a, x0)) > tol) and (it < itmax): # Número de iterações
        if z == 0:
            break
        x0 = x0 - y / z
        y, z = horner(x0, a)
        it+=1
    return x0

def eval_p(a, x):
    '''
        Este é um método usado para avaliar um polinômio em um ponto x, dado seus 
        coeficientes e o valor x. (Visto que precisavamos de uma lista com os 
        coeficientes do polinômio, não podíamos implementar a função da mesma
        forma que a tarefa 2).
    '''
    n = len(a)
    result = 0
    for i in range(n):
        result += (a[i]*(x**i))
    return result

# Define os coeficientes do polinômio
coeficientes = [1, 1, 1, -3, 1]

# Encontra a primeira raiz
# Chute inicial: 1.3
raiz1 = newton_horner(1.3, coeficientes)
print("Aproximação da 1ª raiz:", raiz1)

# Encontra a segunda raiz
# Chute inicial: 2.3
raiz2 = newton_horner(2.3, coeficientes)
print("Aproximação da 2ª raiz:", raiz2)


# - -  Parte Gráfica - -
coeficientes.reverse() #a lista de coeficientes é invertida para seguir o padrão das bibliotecas usadas para formar o gráfico
x = np.linspace(-2, 3, 400)
y_modificado = np.polyval(coeficientes, x) - np.polyval(coeficientes, raiz1) - np.polyval(coeficientes, raiz2)

plt.plot(x, y_modificado, label='p(x)')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(raiz2, linestyle='--', color='green', label='Raiz encontrada a partir do chute inicial')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Polinômio depois do Método Newton-Horner')
plt.legend()
plt.grid(True)
plt.show()