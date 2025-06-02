# Método dos Mínimos Quadrados
import numpy as np

# (a) e (b) -> Aproximando uma função linear

# g = a0 + a1*x
def gl(a: np.ndarray, x: int):
    return a[0] + a[1]*x

# calcula o produto escalar de 2 vetores
def prod_esc(x: np.ndarray, y: np.ndarray):
    soma = 0
    n = len(x)
    for i in range(n):
        soma += x[i]*y[i]
    return soma

# realiza aproximação de uma função linear
def aprox_linear(x: np.ndarray, y: np.ndarray):
    n = len(x)
    ones = np.ones(n)
    M = np.zeros((2,2))
    b = np.zeros(2)
    a = np.zeros(2)
    M[0][0] = prod_esc(ones, ones)
    M[0][1] = prod_esc(ones, x)
    M[1][0] = prod_esc(x, ones)
    M[1][1] = prod_esc(x, x)
    b[0] = prod_esc(ones, y)
    b[1] = prod_esc(x, y)
    a = np.linalg.solve(M, b)
    return a

# (c) Aproximação de uma função por Gauss-Newton

# g = a0 / (1 -  a1*cos(xi))
def g(a: np.ndarray, x: int):
    return a[0]/(1 - a[1]*np.cos(x))
        
# calcula o resíduo, tanto para a função linear quanto a de Gauss-Newton
def R(a: np.ndarray, x: np.ndarray, y: np.ndarray, g: callable):
    soma = 0
    n = len(y)
    for i in range(n):
        r = y[i] - g(a, x[i])
        soma += r**2
    return (soma/2)

# cria matriz-coluna r com os resíduos
def mR(a: np.ndarray, x: np.ndarray, y: np.ndarray):
    n = len(x)
    mR = np.zeros(n)
    for i in range(n):
        mR[i] = y[i] - g(a, x[i])
    return mR

# cria matriz J com derivadas parciais
def J(x: np.ndarray, a: np.ndarray):
    n = len(x)
    Jac = np.zeros((n,2))
    for i in range(n):
        Jac[i][0] = 2*(g(a, x[i]))*(-1/( -a[1]*np.cos(x[i]) + 1))
        Jac[i][1] = 2*(g(a, x[i]))*(-a[0]*np.cos(x[i])/(-a[1]*np.cos(x[i]) + 1)**2)
    return Jac

# Método de Gauss-Newton
def Gauss_Newton(x: np.ndarray, y: np.ndarray, a0: np.ndarray, tol: float):
    itmax = 200
    it = 1
    erro = 1
    n = len(a0)
    a = np.zeros(n)
    while erro > tol and it < itmax:
       r = mR(a0, x, y)
       Ja = J(x, a0)
       Jt = np.transpose(Ja)
       s = np.linalg.solve(np.matmul(Jt,Ja),np.matmul(Jt, r))
       a = a0 - s
       erro = max(abs(s))
       a0 = a
       it += 1
    return a

# função que compara resultados das 2 aproximaçÕes
def comparar_resultados(a1: np.ndarray, a2: np.ndarray, x: np.ndarray, y: np.ndarray):
   print("COMPARANDO RESULTADOS")
   print("Linearização:")
   print(f"a0 = {a1[0]}")
   print(f"a1 = {a1[1]}")
   print(f"Resíduo: {R(a1, x, y, gl)}")
   print("Gauss-Newton:")
   print(f"a0 = {a2[0]}")
   print(f"a1 = {a2[1]}")
   print(f"Resíduo: {R(a2, x, y, g)}")

    
# parâmetros pras funções
a0 = np.array([2,1])
x = np.radians(np.array([48, 67, 83, 108, 206]))
y = np.array([2.70, 2.00, 1.61, 1.20, 1.02])
tol = 1e-6

# armazena resultados
a1 = aprox_linear(x, y)
a2 = Gauss_Newton(x, y, a0, tol)

# compara resultados
comparar_resultados(a1, a2, x, y)