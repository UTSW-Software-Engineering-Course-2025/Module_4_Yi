%% pipeline_CaSpEr_SWE_template.m

%% Parameters (Adjust paths according to your environment)

inputPath0 = "."; 
inputPath1 = "./raw";

outputPath1 = "./output";
outputDir = fullfile(inputPath0, outputPath1, "HMM_template");
if ~isfolder(outputDir); mkdir(outputDir); end

% Number of hidden states
numStates = 5;
M = numStates-1;
BIC_k = (M+1)*(M-1) + 3*M;      % a degree of freedom of HMM or the num of params
                                        
%% Input

% .csv file with caclium activity time courses of 206 neurons
inputfname = "actSig_HCLindexed.csv";

% numNeuron * nFrames = nRow * nCol
actmap = readmatrix(fullfile(inputPath0, inputPath1, inputfname)); 
%actmap = actmap(1,:);

% Visualize neuron Ca activities
q = quantile(actmap(:), [0.005, 0.995]);
famap = figure('Position', [100 100 1200 400]);
imagesc(actmap, [q(1), q(2)]), colormap(jet), colorbar 

%% saveas

saveas(famap, fullfile(outputDir, 'actmap.png'), 'png')
%saveas(famap, fullfile(outputDir, 'actmap.fig'), 'fig')

%% Output

numFrames = size(actmap, 2);
numNeurons = size(actmap, 1);

hmm_statemap = zeros(numNeurons, numFrames);
hmm_binarymap = zeros(numNeurons, numFrames);
hmm_logL = zeros(numNeurons, 1);
hmm_BIC = zeros(numNeurons, 1);

% to save AR-HMM coefficients
arhmm_coef_intercept = zeros(numNeurons, numStates - 1);
arhmm_coef_AR1coef = zeros(numNeurons, numStates - 1);
arhmm_coef_sigmasq = zeros(numNeurons, numStates - 1);

% numStates = 2 for the case of no Ca spike (AR(1) with one state)
oneState_BIC = zeros(numNeurons, 1);

%% One state modeling to check if no spike happens

tic     % parfor
for nid = 1:size(actmap, 1)
    ca = actmap(nid, :);
    
    % Jordan decomposition
    ca1 = lagmatrix(ca, 1);
    ca1 = reshape(ca1, 1, []);
    ca1(1) = ca(1);         
    diffs = ca - ca1;
    indicP = (diffs > 0);
    indicM = (diffs <= 0);
    posDiff = max(0, diffs);

    % Define cumulative sums of positive increments ((x)+)
    y = cumsum(posDiff);      

    % Exclude a flat period of y
    x = y(indicP);
    x1 = lagmatrix(x, 1);
    x1 = reshape(x1, 1, []);
    x1(1) = 0;
    
    % least square estimates of AR1
    mu = mean(x); mu1 = mean(x1);
    phi1 = mean( (x(:) - mu) .* (x1(:) - mu1) ) ./ mean( (x1(:) - mu1).^2 );
    phi0 = mu - phi1 * mu1;
    residuals = x(:) - phi0 - phi1 .* x1(:);
    sigmasq = mean( residuals.^2 );
    
    % logLik and BIC
    T = length(x);
    logL_oneState = (-T/2) * log(sigmasq) - T/2 * log(2*pi) - T/2;
    oneState_BIC(nid) = -2 * logL_oneState + BIC_k * log(T);    
        
end
toc

%% HMM fitting using EM algorithm 
% (eg neuron, i=140 ==> responding to E+P)
% running time = 30 sec

tic      
for nid = 1:size(actmap, 1) 

    ca = actmap(nid, :);
    %f1 = figure('Position', [100 100 1600 400]);
    %plot(ca)

    %% EM algo for HMM

    [fittedStates, initPr, tranPr, intercepts, AR1coef, sigmasq, deltaMat, logL, T] = ...
                    hmmCaSpikes_template(ca, numStates);

    %% output

    %ca1 = lagmatrix(ca, 1);
    %ca1 = reshape(ca1, 1, []);
    %ca1(1) = ca(1);         
    %diffs = ca - ca1;
    %posDiff = max(0, diffs);

    % checking
    %disp('==== States summary')
    %splitapply(@mean, posDiff, fittedStates')
    %tabulate(fittedStates)

    % per-neuron output
    hmm_statemap(nid, :) = fittedStates';
    hmm_binarymap(nid, :) = double(fittedStates == numStates)';
    hmm_logL(nid) = logL;    
    hmm_BIC(nid) = -2 * logL + BIC_k * log(T);

    %disp(['=== k: ', num3str(k)])
    %disp('==== Intercept:')
    %disp(intercepts)
    %disp('==== AR1 coef:')
    %disp(AR1coef)
    %disp('==== Sigma_squared')
    %disp(sigmasq)

    arhmm_coef_intercept(nid, :) = intercepts;
    arhmm_coef_AR1coef(nid, :) = AR1coef;
    arhmm_coef_sigmasq(nid, :) = sigmasq;

end
toc

%% save

writematrix(hmm_statemap, fullfile(outputDir, 'hmm_statemap.csv'));
writematrix(hmm_binarymap, fullfile(outputDir, 'hmm_binarymap.csv'));
writematrix(hmm_logL, fullfile(outputDir, 'hmm_logL.csv'));

writematrix(arhmm_coef_intercept, fullfile(outputDir, 'arhmm_coef_intercept.csv'));
writematrix(arhmm_coef_AR1coef, fullfile(outputDir, 'arhmm_coef_AR1coef.csv'));
writematrix(arhmm_coef_sigmasq, fullfile(outputDir, 'arhmm_coef_sigmasq.csv'));

%% optimal num of hidden states selection step (skipped)
% In this example, we can check if oneState_BIC > hmm_BIC or if K=5 is
% better than K=1 for individual neurons.

disp("==== Is the multi-state model better than one state model (no spike) for all neurons?")
disp(all(oneState_BIC > hmm_BIC))

%% plot binary map

% === 全程 binary map 可视化 ===
fs_t = figure('Position', [100 100 1200 400]);
imagesc(hmm_binarymap)
colormap(gray), colorbar                  
title('Full binary spike map');
xlabel('Time frame'); ylabel('Neuron index');
set(gca, 'FontSize', 12);
saveas(fs_t, fullfile(outputDir, 'hmm_binarymap_full.png'), 'png');

fs1 = figure('Position', [100 100 1200 800]);
imagesc(hmm_binarymap(:, 1:750))    
colormap(gray), colorbar

fs2 = figure('Position', [100 100 1200 800]);
imagesc(hmm_binarymap(:, 751:1500))    
colormap(gray), colorbar

% binary map visualiziation is not accurate due to the big size

tmpmap = imresize(hmm_binarymap, [1600, 4800], 'nearest');

fs3 = figure('Position', [100 100 1200 400]);
h = imagesc(logical(tmpmap));
colormap(gray), colorbar
set(gca, 'YTick', []);

saveas(fs3, fullfile(outputDir, 'hmm_binarymap0.png'), 'png')
%saveas(fs3, fullfile(outputDir, 'hmm_binarymap0.fig'), 'fig')

%
saveas(fs1, fullfile(outputDir, 'hmm_binarymap1.png'), 'png')
%saveas(fs1, fullfile(outputDir, 'hmm_binarymap1.fig'), 'fig')
saveas(fs2, fullfile(outputDir, 'hmm_binarymap2.png'), 'png')
%saveas(fs2, fullfile(outputDir, 'hmm_binarymap2.fig'), 'fig')

%% plot single TS
% running time = ~2min

tic
for nid = 1:size(actmap, 1)

    ca = actmap(nid, :);
    fittedStates = hmm_statemap(nid, :);


    f2 = figure('Position', [100 100 1200 400], 'Visible', 'off');
    plot(ca)
    hold on
    colPal = jet(numStates+2); 
    s = 5;
    idx = find(fittedStates == s);
    scatter(idx, ca(idx), [], 'red');

    saveas(f2, fullfile(outputDir, ['onlySpike_neuronTS_', num2str(nid), '.png']), 'png')
    %saveas(f2, fullfile(outputDir, ['onlySpike_neuronTS_', num2str(nid), '.fig']), 'fig')

end
toc

%% plot single TS with all states annotated
% running time = ~2min

for nid = 1:size(actmap, 1)
    
    ca = actmap(nid, :);
    fittedStates = hmm_statemap(nid, :);

    f3 = figure('Position', [100 100 1200 400], 'Visible', 'off');
    plot(ca)
    hold on
    colPal = jet(numStates+2); 
    sc = cell(1, numStates);
    for s = 1:numStates
    %s = 5;
    idx = find(fittedStates == s);
    sc{s} = scatter(idx, ca(idx), 16, 'MarkerEdgeColor', colPal(s+1, :));
    end
    legend([sc{:}], split(num2str(1:numStates)))

    saveas(f3, fullfile(outputDir, ['allStates_neuronTS_', num2str(nid), '.png']), 'png')
    %saveas(f3, fullfile(outputDir, ['allStates_neuronTS_', num2str(nid), '.fig']), 'fig')

end 

%% EOF