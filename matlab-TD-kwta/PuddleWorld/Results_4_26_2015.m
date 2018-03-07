
%% Results

numTotalTests = 30;

nMeshx = 20; nMeshy = 20;
nTilex = 1; nTiley = 1;
% parameter of Gaussian Distribution
sigmax = 1.0 / nMeshx; 
sigmay = 1.0 / nMeshy;

s_end = [1.0,1.0];

% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;
% smoother state space with tiling
xgrid = 1 / (nMeshx * nTilex);
ygrid = 1 / (nMeshy * nTiley);
xVector = 0:xgrid:1;
yVector = 0:ygrid:1;

grafica = false;
withBias = false; % it's a input to functions; should be declared!

functionApproximator = 'Qtable';
filename = strcat('variables-test','-',functionApproximator,'-','Mesh',num2str(nMeshx),'.mat');  
load(filename,'data','weights','nMeshx','nMeshy','nTilex','nTiley','convergence');
[Qtrue] = plotQ_PolicyMap(s_end,nMeshx,xVector,xInputInterval,nMeshy,yVector,yInputInterval,weights,functionApproximator,grafica);
trueDiscReward = zeros(length(xVector),length(yVector));
totalRunNotinPuddle = 0;
for i=1:length(xVector),
        for j=1:length(yVector),
            s_0 = [xVector(i),yVector(j)];
            [agentinPuddle,~] = CreatePuddle(s_0);
            if ~agentinPuddle,
                trueDiscReward(i,j) = plotPath2Goal(s_0,s_end,xgrid,nMeshx,xVector,xInputInterval,sigmax,nMeshy,ygrid,yVector,yInputInterval,sigmay,weights,functionApproximator,grafica);
                totalRunNotinPuddle = totalRunNotinPuddle+1; 
            end
        end
end


grafica = false;
withBias = false;
counter = 1;
discRewVector = zeros(1,numTotalTests);
RatiosuccessfulRun = zeros(1,numTotalTests);
error_MSE_vector = zeros(1,numTotalTests);

id = [2,5,8,9,10,11,13,22,24,29];


%% All hidden units participate in forward path and All hidden units receive error signal
%% shunt = 1
functionApproximator = 'allHiddenUnitsForwardAndAllGetErrorsShunt1';
estimateDiscRew = zeros(length(xVector),length(yVector));
error_reward_newMatrix = zeros(length(xVector),length(yVector));
successfulRun = 0;
for testi=1:numTotalTests,
    successfulRun = 0;
    filename = strcat('variables-test',num2str(testi),'-',functionApproximator,'Mesh',num2str(nMeshx),'-','Tile',num2str(max(nTilex,nTiley)),'.mat');  
    load(filename,'data','weights','nMeshx','nMeshy','nTilex','nTiley','convergence');
    Qestimate = plotQ_PolicyMap(s_end,nMeshx,xVector,xInputInterval,nMeshy,yVector,yInputInterval,weights,functionApproximator,grafica);    
    if ~ismember(testi,id),
    for i=1:length(xVector),
        for j=1:length(yVector),
            s_0 = [xVector(i),yVector(j)];
            [agentinPuddle,~] = CreatePuddle(s_0);
            if ~agentinPuddle,
                [estimateDiscRew(i,j),agentReached2Goal] = plotPath2Goal(s_0,s_end,xgrid,nMeshx,xVector,xInputInterval,sigmax,nMeshy,ygrid,yVector,yInputInterval,sigmay,weights,functionApproximator,grafica);
                if agentReached2Goal,
                    error_reward_newMatrix(i,j) = estimateDiscRew(i,j) - trueDiscReward(i,j);
                    successfulRun = successfulRun + 1;
                else
                    error_reward_newMatrix(i,j) = 0;
                end
            end
        end
    end
    
    discRewVector(counter) = norm(estimateDiscRew-trueDiscReward,2);
    RatiosuccessfulRun(counter) = successfulRun / totalRunNotinPuddle;
    error_MSE_vector(counter) = mse(error_reward_newMatrix);
    counter = counter + 1;
    end
end



fprintf('Error of MSE for states outside of puddle \n');
fprintf('%12s %12s \n','test number:','Allf-Allb');
for testi=1:20,
    fprintf('%12d %12.3f  \n',testi,error_MSE_vector(testi));
end
fprintf('\n');

fprintf('Mean of MSE \n');
fprintf('%12s %12s %12s %12s %12s \n','test type:','Winf-Winb','Allf-Winb','Winf-Allb','Allf-Allb');
fprintf('%12s %12.3f %12.3f %12.3f %12.3f   \n','',mean(error_MSE_vector(1:20)))

fprintf('\n');

fprintf('std of MSE \n');
fprintf('%12s %12s \n','test type:','Allf-Allb');
fprintf('%12s %12.3f \n','',std(error_MSE_vector(1:20)));
fprintf('\n');

fprintf('Success Ratio \n');
fprintf('%12s %12s   \n','test number:','Allf-Allb');
for testi=1:20,
    fprintf('%12d %12.3f \n',testi,RatiosuccessfulRun(testi));
end
fprintf('\n');

fprintf('Mean of success \n');
fprintf('%12s %12s %12s %12s %12s \n','test type:','Winf-Winb','Allf-Winb','Winf-Allb','Allf-Allb');
fprintf('%12s %12.3f %12.3f %12.3f %12.3f   \n','',mean(RatiosuccessfulRun(1:20)))

fprintf('\n');


fprintf('std of MSE \n');
fprintf('%12s %12s \n','test type:','Allf-Allb');
fprintf('%12s %12.3f \n','',std(RatiosuccessfulRun(1:20)));
fprintf('\n');


