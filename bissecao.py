import numpy as np
def calc_alfa(d, x, calc_grad):
    alfa_u = np.random.rand()
    xn = x + alfa_u * d
    g = calc_grad(xn)
    hl = np.dot(g, d)

    alfa_l = 0.0
    while hl < 0:
        alfa_l = alfa_u
        alfa_u *= 2
        xn = x + alfa_u * d
        g = calc_grad(xn)
        hl = np.dot(g, d)

    epsilon = 1e-3
    kmax = int(np.ceil(np.log2((alfa_u - alfa_l) / epsilon)))
    it, itmax = 0, 20

    while it < min(kmax, itmax) and abs(hl) > 1e-4:
        it += 1
        alfa_m = 0.5 * (alfa_l + alfa_u)
        xn = x + alfa_m * d
        g = calc_grad(xn)
        hl = np.dot(g, d)
        if hl > 0:
            alfa_u = alfa_m
        elif hl < 0:
            alfa_l = alfa_m
        else:
            break
    return 0.5 * (alfa_l + alfa_u)

def calc_grad(x):
    return 2 * x

x = np.array([2.0])   # ponto inicial
d = -calc_grad(x)     # direção de descida

alfa = calc_alfa(d, x, calc_grad)
print(f"Passo ótimo encontrado: {alfa:.4f}")
