# arhmm.py
import numpy as np
from utils import logsumexp

# ---------- 常数 & 工具 ----------
def bic_k(M):
    """自由度 (K= M+1 states; 不含递减状态)"""
    return (M+1)*(M-1) + 3*M

def one_state_bic(yP):
    """单一 AR(1) 模型的 BIC"""
    T     = len(yP)
    yP1   = np.roll(yP,1); yP1[0]=0
    X     = np.vstack([np.ones_like(yP1), yP1]).T
    beta  = np.linalg.lstsq(X, yP, rcond=None)[0]
    resid = yP - X@beta
    sig2  = resid.var()
    logL  = -T/2*np.log(2*np.pi*sig2) - resid.dot(resid)/(2*sig2)
    return -2*logL + bic_k(0)*np.log(T)

# ---------- 发射概率 ----------
def normal_pdf(x, mu, sigma2):
    return (1/np.sqrt(2*np.pi*sigma2))*np.exp(-(x-mu)**2/(2*sigma2))

# ---------- E-步：log-space Forward-Backward ----------
def forward_backward(x, x1, params):
    """对数域实现，返回 γ, ξ, loglik"""
    M, T  = params["M"], len(x)
    phi0, phi1, sig2 = params["phi0"], params["phi1"], params["sigmasq"]
    logA  = np.log(params["A"])
    logpi = np.log(params["pi"])

    # 发射 log-prob
    logB = np.empty((T,M))
    for j in range(M):
        mu     = phi0[j] + phi1[j]*x1
        var    = sig2[j]
        logB[:,j] = -0.5*((x-mu)**2)/var -0.5*np.log(2*np.pi*var)

    # Forward
    log_alpha = np.empty_like(logB)
    log_alpha[0] = logpi + logB[0]
    for t in range(1,T):
        log_alpha[t] = logsumexp(log_alpha[t-1][:,None] + logA, axis=0) + logB[t]

    # Backward
    log_beta  = np.zeros_like(logB)
    for t in range(T-2,-1,-1):
        tmp = logA + (logB[t+1] + log_beta[t+1])[None,:]
        log_beta[t] = logsumexp(tmp, axis=1)

    loglik = logsumexp(log_alpha[-1], axis=0)
    log_gamma = log_alpha + log_beta - loglik
    gamma     = np.exp(log_gamma)

    # ξ
    xi = np.empty((T-1,M,M))
    for t in range(T-1):
        tmp = (log_alpha[t][:,None] + logA +
               logB[t+1][None,:] + log_beta[t+1][None,:]) - loglik
        xi[t] = np.exp(tmp)
    return gamma, xi, loglik

def forward_backward_scaling(x, x1, params):
    """
    Forward-backward with scaling (linear space) version to match the API of `forward_backward`.
    Args:
        x:     (T,) observed sequence
        x1:    (T,) lag-1 version of x, where x1[0]=0
        params: dictionary with keys:
                - M: number of states
                - pi: (M,) initial probabilities
                - A:  (M,M) transition matrix
                - phi0, phi1, sigmasq: (M,) AR(1) parameters
    Returns:
        gamma: (T, M) posterior state probabilities
        xi:    (T-1, M, M) pairwise transition probabilities
        loglik: float, log-likelihood of the observed sequence
    """
    T, M = len(x), params["M"]
    pi, A = params["pi"], params["A"]
    phi0, phi1, sigma2 = params["phi0"], params["phi1"], params["sigmasq"]

    alpha = np.zeros((T, M))
    beta = np.zeros((T, M))
    scale = np.zeros(T)

    # emission probabilities
    B = np.zeros((T, M))
    for j in range(M):
        mu = phi0[j] + phi1[j] * x1
        std = np.sqrt(sigma2[j])
        B[:, j] = (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mu) ** 2) / sigma2[j])

    # Forward
    alpha[0] = pi * B[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]

    for t in range(1, T):
        for j in range(M):
            alpha[t, j] = B[t, j] * np.sum(alpha[t - 1] * A[:, j])
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]

    # Backward
    beta[-1] = 1.0 / scale[-1]
    for t in reversed(range(T - 1)):
        for i in range(M):
            beta[t, i] = np.sum(A[i, :] * B[t + 1, :] * beta[t + 1])
        beta[t] /= scale[t]

    # gamma
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    # xi
    xi = np.zeros((T - 1, M, M))
    for t in range(T - 1):
        denom = np.sum(alpha[t][:, None] * A * B[t + 1] * beta[t + 1])
        for i in range(M):
            xi[t, i, :] = alpha[t, i] * A[i, :] * B[t + 1, :] * beta[t + 1, :]
        xi[t] /= denom

    loglik = np.sum(np.log(scale))

    return gamma, xi, loglik


# ---------- M-步 ----------
def m_step(x, x1, gamma, xi):
    T,M = gamma.shape
    pi  = gamma[0]

    A   = xi.sum(0) / gamma[:-1].sum(0)[:,None]
    A   = A / A.sum(1, keepdims=True)   # 防数值漂

    phi0 = np.empty(M); phi1 = np.empty(M); sig2 = np.empty(M)
    for j in range(M):
        w     = gamma[:,j]
        W     = w.sum()
        xbar  = (w*x).sum()/W
        x1bar = (w*x1).sum()/W
        Sxx   = (w*(x1-x1bar)**2).sum()
        Sxy   = (w*(x1-x1bar)*(x-xbar)).sum()
        phi1[j]= Sxy/(Sxx+1e-12)
        phi0[j]= xbar - phi1[j]*x1bar
        resid  = x - (phi0[j] + phi1[j]*x1)
        sig2[j]= (w*resid**2).sum()/W + 1e-12  # 加 ε 防 0
    return dict(M=M, pi=pi, A=A, phi0=phi0, phi1=phi1, sigmasq=sig2)

# ---------- EM 主函数 ----------
def em_arhmm(x, x1, M=4, max_iter=500, tol=1e-9):
    np.random.seed(0)
    params = dict(
        M        = M,
        pi       = np.ones(M)/M,
        A        = np.random.rand(M,M),
        phi0     = np.random.randn(M),
        phi1     = np.random.rand(M)*0.5,
        sigmasq  = np.ones(M)
    ); params['A'] /= params['A'].sum(1,keepdims=True)

    ll_prev = -np.inf
    for it in range(max_iter):
        # gamma, xi, ll = forward_backward(x, x1, params)
        gamma, xi, ll = forward_backward_scaling(x, x1, params)
        if np.abs((ll-ll_prev)/np.abs(ll_prev+1e-12)) < tol:
            break
        params = m_step(x,x1,gamma,xi)
        ll_prev = ll
    states_soft = gamma.argmax(1)        # soft 最大分配
    return states_soft, ll, params

# ---------- Viterbi 解码 ----------
def viterbi_path(x, x1, params):
    M = params["M"]; T=len(x)
    phi0,phi1,sig2 = params["phi0"],params["phi1"],params["sigmasq"]
    logA  = np.log(params["A"]);  logpi = np.log(params["pi"])

    logB = np.empty((T,M))
    for j in range(M):
        mu=phi0[j]+phi1[j]*x1; var=sig2[j]
        logB[:,j]=-0.5*((x-mu)**2)/var -0.5*np.log(2*np.pi*var)

    delta = np.empty_like(logB); psi=np.empty_like(logB,dtype=int)
    delta[0]=logpi+logB[0]
    for t in range(1,T):
        tmp = delta[t-1][:,None] + logA
        psi[t]=tmp.argmax(0); delta[t]=tmp.max(0)+logB[t]
    path=np.empty(T,dtype=int); path[-1]=delta[-1].argmax()
    for t in range(T-2,-1,-1):
        path[t]=psi[t+1,path[t+1]]
    return path
