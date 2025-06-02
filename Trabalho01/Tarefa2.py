from math import sqrt as nsqrt
from cmath import sqrt as csqrt

def mullers_method(f: callable, x0: float, x1: float, x2: float)  -> float:

    '''
        O Método de Muller é usado para aproximar as raizes de uma função f.
        Dado 3 chutes iniciais (x0, x1 e x2), cálcula o próximo chute como sendo 
        o zero da parábola que passa por (x0,f(x0)), (x1,f(x1)) e (x2,f(x2)).
    '''

    tol = 0.000000000000001 #tolerância máxima de erro da função
    itmax = 100 #número máximo de iterações
    x = x2
    it = 2

    while ( (abs(f(x)) > tol) and (it < itmax) ):
        c = f(x2)
        q0 = (f(x0) - f(x2))/(x0 - x2)
        q1 = (f(x1) - f(x2))/(x1 - x2)
        a = (q0 - q1)/(x0 - x1)
        b = q0*(x2-x1)/(x0 - x1) + q1*(x0-x2)/(x0-x1)
        x = x2 - (2*c/(b + b/abs(b)*sqrt(b**2 - 4*a*c)))
        x0 = x1
        x1 = x2
        x2 = x
        it +=1
    
    return x

def sqrt(x: float): 
    
    '''
        Esta função é usada dentro do Método de Muller para decidir 
        se iremos tirar a raiz de um número real (math.sqrt()) ou 
        complexo não-real(cmath.sqrt()).
    '''

    if isinstance(x, complex) or x < 0:
        return csqrt(x)
    else:
        return nsqrt(x)

def test_function(x: float):
    '''
        Esta é a função usada nos testes, dada pela Professora no enunciado.
    '''
    return x**4 - 3*x**3 + x**2 + x + 1

print('Aproximação da 1ª raiz real: %.14f' % mullers_method(test_function, 1.3, 1.4, 1.5))

print('Aproximação da 2ª raiz real: %.14f' % mullers_method(test_function, 2.2, 2.3, 2.4))

print(f'Aproximação de uma raiz complexa: {mullers_method(test_function, -0.5, 0, 0.5):.14f}')