import numpy as np
import matplotlib.pyplot as plt

# Variáveis globais
tol = 1e-6
kmax = 20

# Matrizes do Sistema 1
def F1(x: np.ndarray):
    n = len(x)
    F = np.zeros(n)
    F[0] = x[0]**2 - x[1]**2 - 1
    F[1] = 2 * x[0] * x[1]
    return F

def J1(x: np.ndarray):
    n = len(x)
    J = np.zeros((n, n))
    J[0][0] = 2 * x[0]  # df0/dx0
    J[0][1] = -2 * x[1]  # df0/dx1
    J[1][0] = 2 * x[1]  # df1/dx0
    J[1][1] = 2 * x[0]  # df1/dx1
    return J

# Matrizes do Sistema 2
def F2(x: np.ndarray):
    n = len(x)
    F = np.zeros(n)
    F[0] = x[0]**3 - 3*x[0]*(x[1]**2) - 1
    F[1] = 3*(x[0]**2)*x[1] - x[1]**3
    return F

def J2(x: np.ndarray):
    n = len(x)
    J = np.zeros((n, n))
    J[0][0] = 3 * (x[0]**2) - 3 * (x[1]**2)  # df0/dx0
    J[0][1] = -6 * x[0] * x[1]  # df0/dx1
    J[1][0] = 6 * x[0] * x[1]  # df1/dx0
    J[1][1] = 3 * (x[0]**2) - 3 * (x[1]**2)  # df1/dx1
    return J

# Método de Newton para achar solução do sistema
def Newton(x0: np.ndarray, F: callable, J: callable):
    n = len(x0)
    x = np.zeros(n)
    erro = 1  # tamanho do erro
    it = 0  # quantidade de iterações
    Fx = F(x0)
    resmax = max(abs(Fx))
    while (resmax > tol or erro < tol) and it < kmax:
        Jx = J(x0)
        s = np.linalg.solve(Jx, -Fx)
        x = x0 + s
        erro = max(abs(s))
        Fx = F(x)
        resmax = max(abs(Fx))
        x0 = x
        it += 1
    return x, it

# Função para gerar os mapas de convergência
def gerar_mapas(F, J, solucoes, T, nome_sistema):
    N = T.shape[0]
    mapa_convergencia = np.zeros((N, N))
    mapa_iteracoes = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            x0 = T[i, j]
            solucao, iteracoes = Newton(x0, F, J)
            erro_min = np.inf
            solucao_index = -1
            for k, s in enumerate(solucoes):
                erro = np.linalg.norm(solucao - s, ord=np.inf)
                if erro < erro_min:
                    erro_min = erro
                    solucao_index = k
            if erro_min < tol:
                mapa_convergencia[i, j] = solucao_index + 1
            else:
                mapa_convergencia[i, j] = 0
            mapa_iteracoes[i, j] = iteracoes

    # Plotar os mapas de convergência
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(mapa_convergencia, extent=(T[0,0,0], T[-1,0,0], T[0,0,1], T[0,-1,1]), origin='lower', cmap='tab10')
    plt.colorbar()
    plt.title(f'Mapa de Convergência - {nome_sistema}')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.imshow(mapa_iteracoes, extent=(T[0,0,0], T[-1,0,0], T[0,0,1], T[0,-1,1]), origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(f'Mapa de Iterações - {nome_sistema}')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

# Definir a grade T para o sistema 1
N = 500
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
T1 = np.array([[[xi, yi] for yi in y] for xi in x])

# Soluções do sistema 1
solucoes1 = [np.array([1, 0]), np.array([-1, 0])]

# Gerar e plotar os mapas para o sistema 1
gerar_mapas(F1, J1, solucoes1, T1, 'Sistema 1')

# Definir a grade T para o sistema 2
x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1.5, 1.5, N)
T2 = np.array([[[xi, yi] for yi in y] for xi in x])

# Soluções do sistema 2
solucoes2 = [np.array([1, 0]), np.array([-0.5, -np.sqrt(3)/2]), np.array([-0.5, np.sqrt(3)/2])]

# Gerar e plotar os mapas para o sistema 2
gerar_mapas(F2, J2, solucoes2, T2, 'Sistema 2')