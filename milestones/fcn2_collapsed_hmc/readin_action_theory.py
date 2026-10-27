"""
Self-consistent read-in action for a two-layer erf network, target y = He1 + eps*He3 (normalized He3).

Model (mean-field, He1-He3 block, perp norm frozen):
    features      c1(w) = (2/sqrt(pi)) w / sqrt(T+2w^2),   c3(w) = -sqrt(8/(3pi)) w^3 / (T+2w^2)^(3/2)
    action        v~(w) = w^2 - (D . c(w))^2,    V(w) = beta * v~(w),   beta = d/(2 sigma_w^2)
    kernel        J_mn = < c_m c_n >  with weight exp(-V(w))   (exact 1D quadrature over all wells)
    closed loop   (I + J(D)/lam) D = sqrt(Chat) * (1, eps)
Reduced parameters:
    T    = 1 + 2 sigma_w^2 (d-1)/d
    lam  = kappa / (P sigma_A^2)                         (paper notation: sigma^2 / (n sigma_a^2))
    Chat = sigma_w^2 sigma_A^2 chi P^2 / (N d kappa^2)   (paper notation: sigma_w^2 * u)
The loop is the gradient condition of a strictly convex function, so the solution is unique;
it is found by Newton's method with backtracking.
"""
import numpy as np

A1, B3 = np.sqrt(2/np.pi), 1/np.sqrt(3*np.pi)          # a, b of the exact r-form


# ---------------------------------------------------------------- model pieces
def reduced_params(N, d, sigma_w2, sigma_A2, P, kappa, chi=1.0):
    """Map raw hyperparameters to (T, beta, lam, Chat)."""
    T = 1 + 2*sigma_w2*(d-1)/d
    beta = d/(2*sigma_w2)
    lam = kappa/(P*sigma_A2)
    Chat = sigma_w2*sigma_A2*chi*P**2/(N*d*kappa**2)
    return T, beta, lam, Chat


def w_grid(wmax=300.0, nw=24001, a=2.0):
    """sinh-spaced grid on w >= 0 (fine near 0, reaches large w) and trapezoid weights."""
    w = a*np.sinh(np.linspace(0, np.arcsinh(wmax/a), nw))
    dw = np.diff(w); tw = np.zeros(nw); tw[:-1] += dw/2; tw[1:] += dw/2
    return w, tw


def features(w, T):
    c1 = 2/np.sqrt(np.pi)*w/np.sqrt(T+2*w*w)
    c3 = -np.sqrt(8/(3*np.pi))*w**3/(T+2*w*w)**1.5
    return c1, c3


def action(w, D, T):
    """Stripped action v~(w) = w^2 - (D.c)^2 (D a 2-vector, w array)."""
    c1, c3 = features(w, T)
    return w*w - (D[0]*c1 + D[1]*c3)**2


def action_r(r, D, T):
    """Exact r-form, r = 2w^2/(T+2w^2): v~ = (T/2) r/(1-r) - r h(r)^2, h = a D1 - b r D3."""
    h = A1*D[0] - B3*r*D[1]
    return T/2*r/(1-r) - r*h*h


# ---------------------------------------------------------------- kernel and its derivatives
class Kernel:
    def __init__(self, T, beta, **grid):
        self.T, self.beta = T, beta
        self.w, self.tw = w_grid(**grid)
        self.c1, self.c3 = features(self.w, T)

    def stats(self, D):
        """Weights, log Z, J = <c c^T>, Cov[(D.c) c] for a single D."""
        g = D[0]*self.c1 + D[1]*self.c3
        V = self.w**2 - g*g
        m = V.min()
        E = np.exp(-self.beta*(V-m))*self.tw
        Z = E.sum(); p = E/Z
        logZ = -self.beta*m + np.log(Z)
        C = np.stack([self.c1, self.c3])                       # 2 x nw
        J = (C*p) @ C.T
        gc = (C*(p*g)).sum(1)
        Cov = (C*(p*g*g)) @ C.T - np.outer(gc, gc)
        return V, p, logZ, J, Cov


# ---------------------------------------------------------------- closed-loop solver
def solve_loop(Chat, lam, eps, K, D0=None, tol=1e-10, maxit=200):
    """Minimize Phi(D) = |D|^2/2 + logZ/(2 beta lam) - sqrt(Chat) y.D  (strictly convex)."""
    z = np.sqrt(Chat)*np.array([1.0, eps])
    D = z.copy() if D0 is None else np.array(D0, float)
    def phi(D):
        return 0.5*D@D + K.stats(D)[2]/(2*K.beta*lam) - z@D
    for _ in range(maxit):
        _, _, _, J, Cov = K.stats(D)
        grad = D + J@D/lam - z
        if np.abs(grad).max() < tol:
            break
        H = np.eye(2) + (J + 2*K.beta*Cov)/lam
        step = -np.linalg.solve(H, grad)
        t, f0 = 1.0, phi(D)
        while phi(D + t*step) > f0 + 1e-4*t*grad@step and t > 1e-12:
            t *= 0.5
        D = D + t*step
    return D, np.abs(grad).max()


# ---------------------------------------------------------------- analysis of a solution
def analyse(D, lam, eps, K):
    V, p, _, J, _ = K.stats(D)
    y = np.array([1.0, eps])
    fbar = J @ np.linalg.solve(J + lam*np.eye(2), y)         # posterior mean in He1-He3 modes
    Delta = y - fbar                                          # residual
    # minima and basins of v~ on w >= 0
    s = np.sign(np.diff(V)); s[s == 0] = 1
    mins = [i+1 for i in range(len(s)-1) if s[i] < 0 and s[i+1] > 0]
    maxs = [i+1 for i in range(len(s)-1) if s[i] > 0 and s[i+1] < 0]
    centre_min = s[0] > 0
    cuts = [0] + maxs + [len(V)]
    basins = []
    labels = (['centre'] if centre_min else []) + [f'w={K.w[i]:.3f}' for i in mins]
    for (lo, hi), lab in zip(zip(cuts[:-1], cuts[1:]), labels):
        P_b = p[lo:hi].sum()
        contrib = (p[lo:hi]@(K.c1[lo:hi]**2) + p[lo:hi]@(K.c3[lo:hi]**2))/lam
        basins.append(dict(basin=lab, weight=P_b, kernel_share=contrib))
    return dict(D=D, J=J, L1=fbar[0], L3=fbar[1]/eps if eps else np.nan,
                ratio=Delta[1]/Delta[0], basins=basins)


def neg_log_p(w, D, K):
    """-log p(w) up to a constant: beta * v~(w), shifted to min 0 (for comparison with histograms)."""
    V = K.beta*action(np.abs(w), D, K.T)
    return V - V.min()
