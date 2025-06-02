import numpy as np
import matplotlib.pyplot as plt

# Definindo os parâmetros
beta = 10 / (40 * 8 * 24)
gamma = 3 / (15 * 24)

# Condições iniciais
S0 = 49
I0 = 1
R0 = 0
t0 = 0
tn = 25 * 24  # tn em horas

# Passo de tempo
h = 1  # 1 hora

# Definindo as EDOs do modelo SIR
def dS_dt(S, I, R, t):
    return -beta_fifth * S * I

def dI_dt(S, I, R, t):
    return beta_fifth * S * I - gamma * I

def dR_dt(S, I, R, t):
    return gamma * I

# Método de Runge-Kutta de quarta ordem
def runge_kutta(S0, I0, R0, t0, tn, h):
    t = np.arange(t0, tn + h, h)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    S[0], I[0], R[0] = S0, I0, R0
    
    for i in range(1, len(t)):
        k1_S = dS_dt(S[i - 1], I[i - 1], R[i - 1], t[i - 1])
        k1_I = dI_dt(S[i - 1], I[i - 1], R[i - 1], t[i - 1])
        k1_R = dR_dt(S[i - 1], I[i - 1], R[i - 1], t[i - 1])
        
        k2_S = dS_dt(S[i - 1] + k1_S * h / 2, I[i - 1] + k1_I * h / 2, R[i - 1] + k1_R * h / 2, t[i - 1] + h / 2)
        k2_I = dI_dt(S[i - 1] + k1_S * h / 2, I[i - 1] + k1_I * h / 2, R[i - 1] + k1_R * h / 2, t[i - 1] + h / 2)
        k2_R = dR_dt(S[i - 1] + k1_S * h / 2, I[i - 1] + k1_I * h / 2, R[i - 1] + k1_R * h / 2, t[i - 1] + h / 2)
        
        k3_S = dS_dt(S[i - 1] + k2_S * h / 2, I[i - 1] + k2_I * h / 2, R[i - 1] + k2_R * h / 2, t[i - 1] + h / 2)
        k3_I = dI_dt(S[i - 1] + k2_S * h / 2, I[i - 1] + k2_I * h / 2, R[i - 1] + k2_R * h / 2, t[i - 1] + h / 2)
        k3_R = dR_dt(S[i - 1] + k2_S * h / 2, I[i - 1] + k2_I * h / 2, R[i - 1] + k2_R * h / 2, t[i - 1] + h / 2)
        
        k4_S = dS_dt(S[i - 1] + k3_S * h, I[i - 1] + k3_I * h, R[i - 1] + k3_R * h, t[i - 1] + h)
        k4_I = dI_dt(S[i - 1] + k3_S * h, I[i - 1] + k3_I * h, R[i - 1] + k3_R * h, t[i - 1] + h)
        k4_R = dR_dt(S[i - 1] + k3_S * h, I[i - 1] + k3_I * h, R[i - 1] + k3_R * h, t[i - 1] + h)
        
        S[i] = S[i - 1] + (h / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        I[i] = I[i - 1] + (h / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
        R[i] = R[i - 1] + (h / 6) * (k1_R + 2 * k2_R + 2 * k3_R + k4_R)
    
    return t, S, I, R

# Definindo novo valor de beta reduzido em cinco vezes
beta_fifth = beta / 5

# Executando a simulação com beta reduzido em cinco vezes
t_fifth, S_fifth, I_fifth, R_fifth = runge_kutta(S0, I0, R0, t0, tn, h)

# Printando os valores
for i in range(len(t_fifth)):
    print(f"Tempo: {t_fifth[i]:.2f} horas, S: {S_fifth[i]:.2f}, I: {I_fifth[i]:.2f}, R: {R_fifth[i]:.2f}")

# Plotando os resultados da simulação com beta reduzido em cinco vezes
plt.figure(figsize=(14, 8))
plt.plot(t_fifth, S_fifth, 'b', label='S (beta reduzido a 1/5)')
plt.plot(t_fifth, I_fifth, 'r', label='I (beta reduzido a 1/5)')
plt.plot(t_fifth, R_fifth, 'g', label='R (beta reduzido a 1/5)')

plt.xlabel('Tempo (horas)')
plt.ylabel('População')
plt.title('Simulação com Beta Reduzido a 1/5')
plt.legend()
plt.grid(True)
plt.show()
