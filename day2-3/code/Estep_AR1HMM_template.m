function [c, gammaMat, xiArr] = ...
            Estep_AR1HMM_template(x, x1, initPr, tranPr, phi0, phi1, sigmasq, T, M)
% Estep_AR1HMM
%
% J Noh, 2025/02

%% Define objects
gammaMat = zeros(T, M);

%% pdf function values, b_i(x_t | x_(t-1))
bMat = zeros(T, M); 
for i = 1:M
    mu = phi0(i) + phi1(i) * x1(:);
    sigma = ( sigmasq(i) )^0.5;
    bMat(:, i) = pdf('Normal', x(:), mu(:), sigma);
end

%% Forward  α_t(i)
alphaMat = zeros(T, M);
c = zeros(T, 1);              % normalizing scale factor for alpha_t(i)
% t = 1
alphaMat(1, :) = initPr .* bMat(1, :);
c(1)           = sum(alphaMat(1, :));
alphaMat(1, :) = alphaMat(1, :) / c(1);


%% alpha_t(i) forward equation
% t = 2 … T
for t = 2:T
    alphaMat(t, :) = (alphaMat(t-1, :) * tranPr) .* bMat(t, :);  % 行向量
    c(t)           = sum(alphaMat(t, :));
    alphaMat(t, :) = alphaMat(t, :) / c(t);                      % 缩放
end


%% beta_t(j) backward equation
betaMat = zeros(T, M);
betaMat(T, :) = 1 / c(T);        % β_T(i) = 1 ，经缩放后需除以 c(T)
d = zeros(T, 1);              % normalizing scale factor for beta_t(i)

for t = T-1:-1:1
    betaMat(t, :) = ( tranPr .* ( bMat(t+1, :) .* betaMat(t+1, :) ) ) * ones(M,1);
    d(t) = sum(betaMat(t, :));
    betaMat(t, :) = betaMat(t, :)' / d(t);   % 注意转置再缩放
end



%% define gamma_t(i) 
gammaMat = alphaMat .* betaMat;              % 已自动归一化 (∑_i γ_t(i)=1)


%% define xi_t(i,j)  
xiArr = zeros(T-1, M, M);
for t = 1:(T-1)
    numer = ( alphaMat(t, :)' .* tranPr ) .* ( bMat(t+1, :) .* betaMat(t+1, :) );
    denom = sum(numer(:));
    xiArr(t, :, :) = numer / denom;          % (i,j) 元素
end


end
