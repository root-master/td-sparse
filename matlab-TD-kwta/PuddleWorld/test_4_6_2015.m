%% test on date 4/6/2015 on 20 networks 
%% 1) tests - RUN
clc, close all, clear all;
withBias = false;

nMeshx = 20; nMeshy = 20;
nTilex = 1; nTiley = 1;

% functionApproximator = 'allHiddenUnitsForwardButNoErrorForLosers';
% 
% for testCounter=1:20,
%     fprintf('Test run number = %d \n',testCounter);
%     [weights,data,convergence] = main_Continuous(functionApproximator,nMeshx,nMeshy,nTilex,nTiley,withBias);
%     filename = strcat('variables-test',num2str(testCounter),'-',functionApproximator,'Mesh',num2str(nMeshx),'-','Tile',num2str(max(nTilex,nTiley)),'.mat');  
%     save(filename,'data','weights','nMeshx','nMeshy','nTilex','nTiley','convergence');
% end

% functionApproximator = 'LosersForwardZeroButErrorForAll';
% for testCounter=1:20,
%     fprintf('Test run number = %d \n',testCounter);
%     [weights,data,convergence] = main_Continuous(functionApproximator,nMeshx,nMeshy,nTilex,nTiley,withBias);
%     filename = strcat('variables-test',num2str(testCounter),'-',functionApproximator,'Mesh',num2str(nMeshx),'-','Tile',num2str(max(nTilex,nTiley)),'.mat');  
%     save(filename,'data','weights','nMeshx','nMeshy','nTilex','nTiley','convergence');
% end

functionApproximator = 'allHiddenUnitsForwardAndAllGetErrorsShunt1';
% Shunt = 1.0
% for testCounter=21:30,
%     fprintf('Test run number = %d \n',testCounter);
%     [weights,data,convergence] = main_Continuous(functionApproximator,nMeshx,nMeshy,nTilex,nTiley,withBias);
%     filename = strcat('variables-test',num2str(testCounter),'-',functionApproximator,'Mesh',num2str(nMeshx),'-','Tile',num2str(max(nTilex,nTiley)),'.mat');  
%     save(filename,'data','weights','nMeshx','nMeshy','nTilex','nTiley','convergence');
% end

functionApproximator = 'kwtaNN';
shunt = 1.0
for testCounter=1:1,
     fprintf('Test run number = %d \n',testCounter);
     [weights,data,convergence] = main_Continuous(functionApproximator,nMeshx,nMeshy,nTilex,nTiley,withBias);
     filename = strcat('variables-test',num2str(testCounter),'-',functionApproximator,'-shunt1-','Mesh',num2str(nMeshx),'-','Tile',num2str(max(nTilex,nTiley)),'.mat');  
     save(filename,'data','weights','nMeshx','nMeshy','nTilex','nTiley','convergence');
end