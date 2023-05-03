addpath('/Users/ngarnier/Documents/research/entropy/dev/tmp/bin/matlab/');

x=randn(1,1000000);
y=randn(1,1000000);

embed_params = struct('mx',2,'stride',2);

algo_params  = struct('k',5,'algo',1+2,'threads',0);
tic; [I1, I2, std1, std2]=compute_MI(x,y, embed_params, algo_params);
toc;
fprintf(" (single-threaded) MI = %f +/- %f, %f +/- %f\n", I1, std1, I2, std2);

algo_params  = struct('k',5,'algo',1+2,'threads',-1);
tic; [I1, I2, std1, std2]=compute_MI(x,y, embed_params, algo_params); 
toc
fprintf(" (multi-threaded)  MI =%f +/- %f, %f +/- %f\n", I1, std1, I2, std2);

algo_params  = struct('k',5,'algo',1+2,'threads',16*2);
tic; [I1, I2, std1, std2]=compute_MI(x,y, embed_params, algo_params);
toc;
fprintf(" (32 thr. imposed) MI %f +/- %f, %f +/- %f\n", I1, std1, I2, std2);
