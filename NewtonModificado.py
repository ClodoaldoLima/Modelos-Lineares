import numpy as np
import matplotlib.pyplot as plt
import os

def newton_modificado(f_grad, f_hess, x0, tol=1e-6, max_iter=50, lambda0=1e-6, kappa_max=1e8):
    x = x0.astype(float).copy()
    hist = []
    for k in range(max_iter):
        g = f_grad(x)
        H = f_hess(x)
        try:
            cond = np.linalg.cond(H)
        except np.linalg.LinAlgError:
            cond = np.inf
        lambda_reg = lambda0
        H_mod = H.copy()
        if np.isnan(cond) or cond > kappa_max:
            lambda_reg = max(lambda0, (cond / kappa_max) * lambda0)
            H_mod = H + lambda_reg * np.eye(H.shape[0])
        try:
            d = -np.linalg.solve(H_mod, g)
        except np.linalg.LinAlgError:
            d = -g
        x = x + d
        hist.append({'x': x.copy(), 'norm_grad': np.linalg.norm(g), 'cond': cond, 'lambda': lambda_reg})
        if np.linalg.norm(g) < tol:
            break
    return x, hist

def grad_quad(x):
    return np.array([2.0 * x[0] + 4.0])
def hess_quad(x):
    return np.array([[2.0]])

x0 = np.array([10.0])
xmin, hist_quad = newton_modificado(grad_quad, hess_quad, x0, max_iter=50, lambda0=1e-8, kappa_max=1e6)

norms_q = [h['norm_grad'] for h in hist_quad]
conds_q = [h['cond'] for h in hist_quad]
iters_q = np.arange(1, len(hist_quad)+1)

os.makedirs('figuras', exist_ok=True)
plt.figure(figsize=(6,3))
if len(iters_q)>0:
    plt.semilogy(iters_q, norms_q, marker='o')
plt.title('Convergência da norma do gradiente - Quadrática (1D)')
plt.xlabel('Iteração'); plt.ylabel('||grad f|| (escala log)')
plt.grid(True); plt.tight_layout()
plt.savefig('figuras/convergencia_quad.png')
plt.close()

plt.figure(figsize=(6,3))
if len(iters_q)>0:
    plt.plot(iters_q, conds_q, marker='o')
plt.title('Número de condição da Hessiana - Quadrática (1D)')
plt.xlabel('Iteração'); plt.ylabel('cond(H)')
plt.grid(True); plt.tight_layout()
plt.savefig('figuras/condicao_quad.png')
plt.close()

# Rosenbrock
def grad_rosen(x):
    x0, x1 = x[0], x[1]
    d0 = -400*x0*(x1 - x0*x0) - 2*(1 - x0)
    d1 = 200*(x1 - x0*x0)
    return np.array([d0, d1])

def hess_rosen(x):
    x0, x1 = x[0], x[1]
    h00 = -400*(x1 - 3*x0*x0) + 2
    h01 = -400*x0
    h11 = 200
    return np.array([[h00, h01],[h01, h11]])

x0_rosen = np.array([-1.2, 1.0])
xmin_r, hist_rosen = newton_modificado(grad_rosen, hess_rosen, x0_rosen, max_iter=100, lambda0=1e-6, kappa_max=1e8)

norms_r = [h['norm_grad'] for h in hist_rosen]
conds_r = [h['cond'] if np.isfinite(h['cond']) else np.nan for h in hist_rosen]
iters_r = np.arange(1, len(hist_rosen)+1)

plt.figure(figsize=(6,3))
if len(iters_r)>0:
    plt.semilogy(iters_r, norms_r, marker='o')
plt.title('Convergência da norma do gradiente - Rosenbrock (2D)')
plt.xlabel('Iteração'); plt.ylabel('||grad f|| (escala log)')
plt.grid(True); plt.tight_layout()
plt.savefig('figuras/convergencia_rosen.png')
plt.close()

plt.figure(figsize=(6,3))
if len(iters_r)>0:
    plt.plot(iters_r, conds_r, marker='o')
plt.title('Número de condição da Hessiana - Rosenbrock (2D)')
plt.xlabel('Iteração'); plt.ylabel('cond(H)')
plt.grid(True); plt.tight_layout()
plt.savefig('figuras/condicao_rosen.png')
plt.close()

with open('figuras/resumo.txt', 'w') as f:
    f.write('Quadrática (1D) - minimo: {}\n'.format(xmin))
    f.write('Iterações: {}\n'.format(len(hist_quad)))
    f.write('Rosenbrock (2D) - minimo aproximado: {}\n'.format(xmin_r))
    f.write('Iterações: {}\n'.format(len(hist_rosen)))
print('Execução concluída. Figuras salvas em ./figuras/')
