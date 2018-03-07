maxNumEpisodes = 200000;
%% KWTA
functionApproximator = 'kwtaNN';

nMesh = [20,20,20,20];
[convergence,failure,weights,data,nMesh,functionApproximator] = main_Acrobot(functionApproximator,maxNumEpisodes,nMesh);
filename = strcat('Results-Jan-1/',functionApproximator,'-nMesh',num2str(nMesh(1)),'.mat');  
save(filename,'functionApproximator','data','weights','nMesh','convergence','failure');

nMesh = [30,30,30,30];
[convergence,failure,weights,data,nMesh,functionApproximator] = main_Acrobot(functionApproximator,maxNumEpisodes,nMesh);
filename = strcat('Results-Jan-1/',functionApproximator,'-nMesh',num2str(nMesh(1)),'.mat');  
save(filename,'functionApproximator','data','weights','nMesh','convergence','failure');

nMesh = [40,40,40,40];
[convergence,failure,weights,data,nMesh,functionApproximator] = main_Acrobot(functionApproximator,maxNumEpisodes,nMesh);
filename = strcat('Results-Jan-1/',functionApproximator,'-nMesh',num2str(nMesh(1)),'.mat');  
save(filename,'functionApproximator','data','weights','nMesh','convergence','failure');
