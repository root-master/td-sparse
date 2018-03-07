maxNumEpisodes = 200000;

functionApproximator = 'linearNN';
nMesh = [20,20,20,20];
[convergence,failure,weights,data,nMesh,functionApproximator] = main_Acrobot(functionApproximator,maxNumEpisodes,nMesh);
filename = strcat('Results-Jan-1/',functionApproximator,'-nMesh',num2str(nMesh(1)),'.mat');  
save(filename,'functionApproximator','data','weights','nMesh','convergence','failure');
