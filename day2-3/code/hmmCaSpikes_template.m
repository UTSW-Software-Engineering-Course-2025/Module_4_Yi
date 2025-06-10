function [fittedStates, initPr, tranPr, phi0_newlabel, phi1_newlabel, ...
            sigmasq_newlabel, deltaMat, logL, T] = ...
            hmmCaSpikes_template(ca, numStates, emtol)
% Detect calcium spikes based on hiddem Markov modeling
%
% J Noh, 2025/02

%% EM algorithm parameters

maxIter = 5000;
logLik = nan(1, maxIter);   

if nargin < 3
    emtol = 1e-9; 
end

%% Jordan decomposition (f = f+ - f-)

ca1 = lagmatrix(ca, 1);
ca1 = reshape(ca1, 1, []);
ca1(1) = ca(1);         
diffs = ca - ca1;
indicP = (diffs > 0);
indicM = (diffs <= 0);
posDiff = max(0, diffs);

% Define cumulative sums of positive increments (f+)
y = cumsum(posDiff);      

% Exclude a flat period of f+
yP = y(indicP); 
   
yP1 = lagmatrix(yP, 1);
yP1 = reshape(yP1, 1, []);
yP1(1) = 0;

%figure('Position', [100 100 800 400]);
%plot(y)

%figure('Position', [100 100 800 400]);
%negDiff = min(0, diffs);
%y2 = cumsum(negDiff);
%plot(y2)

%% EM 

x = yP;
x1 = yP1;
M = numStates-1;

%% EM (1) initialize

% initial probabilities
initPr = ones(1, M) * 1/M;              % initial state probabilities
tmp = rand(M);                          % uniform(0, 1) distribution
tranPr = tmp ./ sum(tmp, 2);            % transition probabilities

X = [ones(length(x), 1), x1(:)];
phihat = X\x(:);
residuals = x(:) - mtimes(X, phihat);
phi0 = ones(1, M) * phihat(1);                  % AR(1) intercept for each state
phi1 = ones(1, M) * phihat(2);                  % AR(1) coefficient for each state
sigmasq = ones(1, M) * mean(residuals .^ 2);    % AR(1) error variance for each state

%% EM (2) Iterate E- and M-steps
% E-step: compute alpha, beta, gamma, xi

T = length(x);

for k = 1:maxIter
    
    % E-step
    [c, gammaMat, xiArr] = ...
            Estep_AR1HMM_template(x, x1, initPr, tranPr, phi0, phi1, sigmasq, T, M);
    
    % M-step
    tranPr_post = zeros(M, M);                          % initialize

    mu = mean( gammaMat .* x(:) ) ./ mean( gammaMat );
    mu1 = mean( gammaMat .* x1(:) ) ./ mean (gammaMat );
    initPr_post = gammaMat(1, :);

    avgGamma = mean(gammaMat(1:(T-1), :));
    for j = 1:M
        for i = 1:M
            tranPr_post(i, j) = mean( xiArr(:, i, j) ) / avgGamma(i);
        end
    end

    % weighted MLE
    tmp1 = mean( gammaMat .* (x(:) - mu) .* (x1(:) - mu1) );
    tmp2 =  mean( gammaMat .* ((x1(:) - mu1).^2) );
    phi1_post = tmp1 ./ tmp2;

    %phi0_post = mu .* (1 - phi1_post);
    phi0_post = mu - phi1_post .* mu1;
    residualMat = x(:) - phi0_post - phi1_post .* x1(:);

    sigmasq_post = mean( gammaMat .* residualMat.^2 ) ./ mean( gammaMat );  

    %
    initPr = initPr_post;
    tranPr = tranPr_post;
    phi1 = phi1_post;
    phi0 = phi0_post;
    sigmasq = sigmasq_post;

    logLik(k) = sum( log(c(c>0)) );    
    
    if k > 1
        deltaLogLik = abs((logLik(k) - logLik(k-1)) /  logLik(k-1));
        %disp(deltaLogLik)
        if deltaLogLik < emtol; break; end  %disp(k); disp(logLik(k)); break end
    end

end

logL = logLik(k);

%disp(k)
%disp(deltaLogLik)
%disp(logLik(k))

%disp(round(initPr, 3))
%disp(round(tranPr, 3))
%disp('==== Intercept:')
%disp(phi0)
%disp('==== AR1 coeff:')
%disp(phi1)
%disp('==== Sigma_squared:')
%disp(sigmasq)

%% Viterbi

T = length(yP);
M = numStates - 1;
bMat = zeros(T, M); 

% pdf function values, b_i(x_t | x_(t-1))
for i = 1:M
    mu = phi0(i) + phi1(i) * yP1(:);
    sigma = ( sigmasq(i) )^0.5;
    bMat(:, i) = pdf('Normal', yP(:), mu(:), sigma);
end

%tic
[hmmS, deltaMat, ~] = Viterbi_HMM(initPr, tranPr, bMat);
%tabulate(hmmS)
%toc

%% Add the pre-excluded 'ca-activity-decreasing period' and sort states

% relabel hmm-States in a mean-increasing order

diffsP = diffs(indicP);
tmp = nan(1, M);
for i = 1:M
    subvec = diffsP(hmmS == i);
    tmp(i) = nanmean(subvec);
end 

% to sort parameter vectors
[~, ord_sorted] = sort(tmp, 'MissingPlacement', 'first');

%disp(ord_sorted)
%disp(tmp(ord_sorted))

[~, map0] = sort(ord_sorted);
hmmS_newlabel = map0(hmmS)';
% for example, convert the state 'ord_sorted(M)' into newlabel 'M', that
% has the highest mean activity. 
% disp(tmp)
% disp(map0)

% Visualize AR1-HMM fitting directly to f+.
% figure, plot(x), hold on;
% idx = find(hmmS_newlabel == M);
% scatter(idx, x(idx), [], 'red'); 
 
% augment pre-excluded time points where calcium activities are decreasing
fittedStates = zeros(length(ca), 1);
fittedStates(indicM) = 1;                   % activity-decreasing state
fittedStates(indicP) = hmmS_newlabel + 1;   % St2~StK are in the order of 
                                            % increasing mean activity
% to save AR-HMM coefficients
phi0_newlabel = phi0(ord_sorted);
phi1_newlabel = phi1(ord_sorted);
sigmasq_newlabel = sigmasq(ord_sorted);

end
