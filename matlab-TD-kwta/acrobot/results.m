%% Results

%% LOAD
load('./Results-Jan-1/Qtable-nMesh20.mat');
statesList = data.statesList;
Qtable = weights.Qtable;

load('./Results-Jan-1/kwtaNN-nMesh20.mat');
% 
% figure;
% subplot(2,1,1);    
% plot(data.meanDeltaForEpisode)
% ylabel('mean of TD error in a episode')
% title(functionApproximator)    
% subplot(2,1,2); 
% plot(data.steps)
% ylabel('steps')
% 
% figure;
% subplot(2,1,1);    
% plot(data.meanDeltaForEpisodeTest)
% ylabel('mean of TD error in a episode')
% title(functionApproximator)    
% subplot(2,1,2); 
% plot(data.stepsTest)
% ylabel('steps')

grafica = true;
[~,~,capture,error,F] = ...
    runTest(nMesh,functionApproximator,weights,Qtable,statesList,grafica);
% 
% figure
% plot(capture(1,:),'r-'), hold on 
% plot(capture(2,:),'b-'), hold on  
% plot(capture(3,:),'g-'), hold on 
% plot(capture(4,:),'c-'), hold on


load('./Results-Jan-1/regularBPNN-nMesh20.mat');

% figure;
% subplot(2,1,1);    
% plot(data.meanDeltaForEpisode)
% ylabel('mean of TD error in a episode')
% title(functionApproximator)    
% subplot(2,1,2); 
% plot(data.steps)
% ylabel('steps')
% 
% figure;
% subplot(2,1,1);    
% plot(data.meanDeltaForEpisodeTest)
% ylabel('mean of TD error in a episode')
% title(functionApproximator)    
% subplot(2,1,2); 
% plot(data.stepsTest)
% ylabel('steps')

% grafica = true;
% [~,~,capture,F] = ...
%     runTest(nMesh,functionApproximator,weights,Qtable,statesList,grafica);

% figure
% plot(capture(1,:),'r-'), hold on 
% plot(capture(2,:),'b-'), hold on  
% plot(capture(3,:),'g-'), hold on 
% plot(capture(4,:),'c-'), hold on


%load('./Results-Jan-1/linearNN-nMesh20.mat');
% 
% figure;
% subplot(2,1,1);    
% plot(data.meanDeltaForEpisode)
% ylabel('mean of TD error in a episode')
% title(functionApproximator)    
% subplot(2,1,2); 
% plot(data.steps)
% ylabel('steps')
% 
% figure;
% subplot(2,1,1);    
% plot(data.meanDeltaForEpisodeTest)
% ylabel('mean of TD error in a episode')
% title(functionApproximator)    
% subplot(2,1,2); 
% plot(data.stepsTest)
% ylabel('steps')
% 
% grafica = true;
% [~,~,capture,F] = ...
%     runTest(nMesh,functionApproximator,weights,Qtable,statesList,grafica);
% 
% figure
% plot(capture(1,:),'r-'), hold on 
% plot(capture(2,:),'b-'), hold on  
% plot(capture(3,:),'g-'), hold on 
% plot(capture(4,:),'c-'), hold on
% 
% 
