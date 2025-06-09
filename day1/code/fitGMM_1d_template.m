function [labels, mu, sigma2, mixtureProb, membershipProbMat] = fitGMM_1d_template(x, K)
% fitGMM_1d implements 1-dim Gaussian Mixture Modeling.

% Iteration params
maxIter = 500;
logL = nan(1, maxIter);
emtol = 1e-9; 

% Define objects
x = x(:);
N = length(x);
labels = nan(N, 1);
mu = nan(1, K);
sigma2 = nan(1, K);
mixtureProb = nan(1, K); 
l = nan(N, 1);

% Set random initial values
randIdx   = randperm(N, K);
mu_ini        = x(randIdx).';            % 1xK initial means
sigma2    = var(x) * ones(1, K);     % start with global variance
mixtureProb = ones(1, K) / K;        % uniform mixing weights
membershipProbMat = nan(N, K);

% EM iteration
for iter = 1:maxIter
    % M-step
    Nk = sum(membershipProbMat, 1);                 % effective counts (1xK)
    if iter == 1
        Nk = N * mixtureProb;                       % use current π_k
    end
    mu = (membershipProbMat' * x) ./ Nk.';          % 1xK
    if iter == 1
        mu = mu_ini
    % For first iteration fall back to previous μ if γ not defined
    if iter > 1 && any(isnan(mu)), mu = mu_old; end
    mu_old = mu;
    
    % Update variances σ_k² (biased ML estimator)
    for k = 1:K
        diffsq = (x - mu(k)).^2;
        sigma2(k) = (membershipProbMat(:, k)' * diffsq) / Nk(k);
    end
    % Numerical stabilisation
    sigma2 = max(sigma2, 1e-12);
    
    % Update mixture weights π_k
    mixtureProb = Nk / N;    
    
    

    % E-step
    lpx = zeros(N, K);
    for k = 1:K
        lpx(:, k) = log(mixtureProb(k)) + ...
                    (-0.5*log(2*pi*sigma2(k))) + ...
                    (-0.5*(x-mu(k)).^2./sigma2(k));
    end
    
    % Log-sum-exp trick for numerical stability
    maxlog = max(lpx, [], 2);
    lse    = maxlog + log(sum(exp(lpx-maxlog), 2));
    
    % Posterior responsibilities γ_nk
    membershipProbMat = exp(lpx - lse);             % NxK, each row sums to 1
    
    
    
    
    
    % Terminate if converged
    if iter > 1
        deltaLogL = abs((logL(iter) - logL(iter-1)) /  logL(iter-1));
        if deltaLogL < emtol; disp(iter); disp(logL(iter)); break; end
    end
end

% Determine memberships
[~, labels] = max(membershipProbMat, [], 2);








end
