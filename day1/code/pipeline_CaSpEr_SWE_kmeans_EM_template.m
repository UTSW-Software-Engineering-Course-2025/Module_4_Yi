%% pipeline_CaSpEr_SWE_kmeans_EM.m

%% Parameters (Adjust paths according to your environment)

inputPath0 = ".HMM_SWE2025"; 
inputPath1 = "../raw";
                                        
%% Input

% .csv file with caclium activity time courses of 206 neurons
inputfname = "actSig_HCLindexed.csv";

% numNeuron * nFrames = nRow * nCol
actmap = readmatrix(fullfile(inputPath0, inputPath1, inputfname)); 

% Visualize neuron Ca activities
q = quantile(actmap(:), [0.005, 0.995]);
famap = figure('Position', [100 100 1200 400]);
imagesc(actmap, [q(1), q(2)]), colormap(jet), colorbar 

%% kmeans

nid = 140;
ca = actmap(nid, :);
figure('Position', [100 100 1200 400], 'Visible', 'on');
plot(ca)
figure('Position', [100 100 1200 400], 'Visible', 'on');
histogram(ca, 'BinMethod', 'fd')

K = 3;
[labels, centroids] = kmeans(ca', K, 'Replicates', 3);
disp(centroids)
[~, cent_order] = sort(centroids)
[~, mapping] = sort(cent_order)
labels2 = mapping(labels);
tabulate(labels)
tabulate(labels2)

figure('Position', [100 100 1200 400], 'Visible', 'on');
plot(ca)
hold on
colPal = jet(K+3); 
sc = cell(1, K);
for s = 1:K
    idx = find(labels2 == s);
    sc{s} = scatter(idx, ca(idx), 16, 'MarkerEdgeColor', colPal(s+3, :));
end
legend([sc{:}], split(num2str(1:K)))

%% Gaussian Mixutre Model 1-dim 

nid = 140;
ca = actmap(nid, :);
figure('Position', [100 100 1200 400], 'Visible', 'on');
plot(ca)
figure('Position', [100 100 1200 400], 'Visible', 'on');
histogram(ca, 'BinMethod', 'fd')

K = 3;
[labels, mu, sigma2, mixtureProb, membershipProbMat] = fitGMM_1d_template(ca, K);
disp(mu)
[~, mu_order] = sort(mu)
[~, mapping] = sort(mu_order)
labels2 = mapping(labels);
tabulate(labels)
tabulate(labels2)

figure('Position', [100 100 1200 400], 'Visible', 'on');
plot(ca)
hold on
colPal = jet(K+3); 
sc = cell(1, K);
for s = 1:K
    idx = find(labels2 == s);
    sc{s} = scatter(idx, ca(idx), 16, 'MarkerEdgeColor', colPal(s+3, :));
end
legend([sc{:}], split(num2str(1:K)))

%% EOF