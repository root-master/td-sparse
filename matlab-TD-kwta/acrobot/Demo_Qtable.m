maxNumEpisodes = 200000;

functionApproximator = 'Qtable';
nMesh = [20,20,20,20];
[convergence,failure,weights,data,nMesh,functionApproximator] = main_Acrobot(functionApproximator,maxNumEpisodes,nMesh);
filename = strcat('Results-Feb-25/',functionApproximator,'-nMesh',num2str(nMesh(1)),'.mat');  
save(filename,'functionApproximator','data','weights','nMesh','convergence','failure');

maxNumEpisodes = 200000;

functionApproximator = 'Qtable';
nMesh = [25,25,25,25];
[convergence,failure,weights,data,nMesh,functionApproximator] = main_Acrobot(functionApproximator,maxNumEpisodes,nMesh);
filename = strcat('Results-Feb-25/',functionApproximator,'-nMesh',num2str(nMesh(1)),'.mat');  
save(filename,'functionApproximator','data','weights','nMesh','convergence','failure');



