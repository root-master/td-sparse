%function [weights,data,convergence] = sarsa_gridworld(task,functionApproximator,withBias,s_end,weights,data,nMeshx,nTilex,nMeshy,nTiley)
clc, close all, clear all;
withBias = false;

nMeshx = 20; nMeshy = 20;
nTilex = 1; nTiley = 1;

functionApproximator = 'kwtaNN';
shunt = 1.0;

% control task could be 'grid_world' or 'puddle_world'
task = 'puddle_world';
% function approximator can be either 'kwtaNN' or 'regularBPNN'


% goal in continouos state
s_end = [1.0,1.0];

% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;

% the number of states -- This is the gross mesh states ; the 1st tiling 
nStates = length(xInputInterval) * length(yInputInterval); 

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

%% kwta and regular BP Neural Network
% Weights from input (x,y) to hidden layer
InputSize = length(xInputInterval) + length(yInputInterval);
nCellHidden = round(0.5 * nStates);
mu = 0.1;
Wih = mu * (rand(InputSize,nCellHidden) - 0.5);
biasih = mu * ( rand(1,nCellHidden) - 0.5 );
% biasih = zeros(1,nCellHidden);
% Weights from hidden layer to output
Who = mu * (rand(nCellHidden,nActions) - 0.5);
biasho = mu * ( rand(1,nActions) - 0.5 );
% biasho = zeros(1,nActions);

%% Linear Neural Net
mu = 0.1; % amplitude of random weights
Wio = mu * (rand(InputSize,nActions) - 0.5);
biasio = mu * (rand(1,nActions) - 0.5 );

%% Q Table
Qtable = zeros(nStates,nActions);

%% weights
weights = struct;
Qtable = zeros(nStates,nActions);
weights.Wih = Wih;
weights.Who = Who;
weights.biasih = biasih;
weights.biasho = biasho;
weights.biasio = biasio;
weights.Wio = Wio;
weights.Qtable = Qtable;

data = struct;
meanDeltaForEpisode = [];
varianceDeltaForEpisode = [];
stdDeltaForEpisode = [];
data.meanDeltaForEpisode = meanDeltaForEpisode;
data.varianceDeltaForEpisode = varianceDeltaForEpisode;
data.stdDeltaForEpisode = stdDeltaForEpisode;
initialWinners = [];
ultimateWinners = [];
data.initialWinners = initialWinners;
data.ultimateWinners = ultimateWinners;
%% RUN
if withBias,
    wB = 'withBias';
else
    wB = 'withoutBias';
end


Wih = weights.Wih;
biasih = weights.biasih;
Who = weights.Who;
biasho = weights.biasho;
Wio = weights.Wio;
biasio = weights.biasio;
Qtable = weights.Qtable;

initialWinners = data.initialWinners;
ultimateWinners = data.ultimateWinners;

meanDeltaForEpisode = data.meanDeltaForEpisode;
varianceDeltaForEpisode = data.varianceDeltaForEpisode;
stdDeltaForEpisode = data.stdDeltaForEpisode;

alpha = 0.005;
alphaTable = 0.4;

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

gamma = 0.99;    % discounted task 
epsilon = 0.1;  % epsilon greedy parameter

if strcmp(functionApproximator,'Qtable') && isempty(meanDeltaForEpisode), epsilon = 0.1; end

% Max number of iteration in ach episde to break the loop if AGENT
% can't reach the GOAL 
maxIteratonEpisode = 2 * (nMeshx * nTilex + nMeshy * nTiley);


switch functionApproximator,
         case 'kwtaNN'
             shunt = 1.0;
         case 'allHiddenUnitsForwardButNoErrorForLosers'
             shunt = 1.0;
         case 'LosersForwardZeroButErrorForAll'
             shunt = 1.0;
         case 'allHiddenUnitsForwardAndAllGetErrors'
             shunt = 0.0;
         case  'allHiddenUnitsForwardAndAllGetErrorsShunt1'
            shunt = 1.0;
end
             


%%
% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;
% smoother state space with tiling
xgrid = 1 / (nMeshx * nTilex);
ygrid = 1 / (nMeshy * nTiley);
% parameter of Gaussian Distribution
sigmax = 1.0 / nMeshx; 
sigmay = 1.0 / nMeshy;

xVector = 0:xgrid:1;
yVector = 0:ygrid:1;

nStates = length(xInputInterval) * length(yInputInterval);

%% Different Max number of episodes
maxNumEpisodes = 100 * nStates * nTilex * nTiley;

nGoodEpisodes = 0; % a variable for checking the convergence
convergence = false;
agentReached2Goal = false;
agentBumped2wall = false;
%% Episode Loops
ei = 1;
deltaForStepsOfEpisode = [];

while (ei < maxNumEpisodes && ~convergence ), % ei<maxNumEpisodes && % ei is counter for episodes
     
     if ei == 10
        for i=1:length(xVector)
            for j=1:length(yVector)
               xTest = xVector(i);
               yTest = yVector(j);
               xTestInput = sigmax * sqrt(2*pi) * normpdf(xInputInterval,xTest,sigmax);
               yTestInput = sigmay * sqrt(2*pi) * normpdf(yInputInterval,yTest,sigmay);
               stTest = [xTestInput,yTestInput];
               switch functionApproximator,
                    case 'kwtaNN'
                        [~,~,id] = kwta_NN_forward_new(stTest,shunt,Wih,biasih,Who,biasho);
                    case 'allHiddenUnitsForwardButNoErrorForLosers'
                        [~,~,id] = kwta_NN_forward_new(stTest,shunt,Wih,biasih,Who,biasho);
                    case 'LosersForwardZeroButErrorForAll'
                        [~,~,id] = kwta_NN_forward(stTest,shunt,Wih,biasih,Who,biasho);
                    case 'allHiddenUnitsForwardAndAllGetErrors'
                        [~,~,id] = kwta_NN_forward_new(stTest,shunt,Wih,biasih,Who,biasho);
                    case  'allHiddenUnitsForwardAndAllGetErrorsShunt1'
                        [~,~,id] = kwta_NN_forward_new(stTest,shunt,Wih,biasih,Who,biasho);
               end
               initialWinners(i,j,:) = id; 
            end
        end
     end
     
    
     % initialize the starting state - Continuous state
     s = initializeState(xVector,yVector);
     x = s(1); y = s(2);
     switch functionApproximator,
         case 'Qtable',
             [~,idx] = min(dist(x,xInputInterval));
             [~,idy] = min(dist(y,yInputInterval));
             sti = sub2ind([length(xVector),length(yVector)],idx,idy);
     end
     % Gaussian Distribution on continuous state
     xt = sigmax * sqrt(2*pi) * normpdf(xInputInterval,x,sigmax);
     yt = sigmay * sqrt(2*pi) * normpdf(yInputInterval,y,sigmay);
     % Using st as distributed input for function approximator
     st = [xt,yt];     
          
     % initializing time
     ts = 1;
     switch functionApproximator,
         case 'kwtaNN'
             [Q,h,id] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
             
         case 'allHiddenUnitsForwardButNoErrorForLosers'
             [Q,h,id] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
         case 'LosersForwardZeroButErrorForAll'
             [Q,h,id] = kwta_NN_forward(st,shunt,Wih,biasih,Who,biasho);
         case 'allHiddenUnitsForwardAndAllGetErrors'
             [Q,h,id] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
         case  'allHiddenUnitsForwardAndAllGetErrorsShunt1'
             [Q,h,id] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
         case 'regularBPNN',
             [Q,h] = regularBPNN_forward(st, Wih,biasih, Who,biasho);
         case 'linearNN'
              Q  = SimpleNN_forward(st,Wio,biasio);
         case 'Qtable'
              Q = Qtable(sti,:);
     end
     act = e_greedy_policy(Q,nActions,epsilon);

    %% Episode While Loop
    while( ~(s(1)==s_end(1) && s(2)==s_end(2)) && ts < maxIteratonEpisode),
        % update state to state+1
        sp1 = UPDATE_STATE(s,act,xgrid,xVector,ygrid,yVector);
        xp1 = sp1(1); yp1 = sp1(2);
        switch functionApproximator,
         case 'Qtable',
             [~,idxp1] = min(dist(xp1,xInputInterval));
             [~,idyp1] = min(dist(yp1,yInputInterval));
             stp1i = sub2ind([length(xInputInterval),length(yInputInterval)],idxp1,idyp1);
        end
        % PDF of stp1
        xtp1 = sigmax * sqrt(2*pi) * normpdf(xInputInterval,xp1,sigmax);
        ytp1 = sigmay * sqrt(2*pi) * normpdf(yInputInterval,yp1,sigmay);
        stp1=[xtp1,ytp1];
        
        if ( sp1(1)==s_end(1) && sp1(2)==s_end(2) ),
            agentReached2Goal = true;
            agentBumped2wall = false;
        elseif ( sp1(1)==s(1) && sp1(2)==s(2) ),
            agentBumped2wall = true;
            agentReached2Goal = false;
        else
            agentBumped2wall = false;
            agentReached2Goal = false;            
        end
        
        % reward/punishment from Environment
        rew = ENV_REWARD(sp1,agentReached2Goal,agentBumped2wall,nTilex,nTiley);
        switch functionApproximator,
            case 'kwtaNN',
                [Qp1,hp1,idp1] = kwta_NN_forward_new(stp1,shunt,Wih,biasih,Who,biasho);
            case 'allHiddenUnitsForwardButNoErrorForLosers'
                [Qp1,hp1,idp1] = kwta_NN_forward_new(stp1,shunt,Wih,biasih,Who,biasho);
            case 'LosersForwardZeroButErrorForAll'
                [Qp1,hp1,idp1] = kwta_NN_forward(stp1,shunt,Wih,biasih,Who,biasho);
            case 'allHiddenUnitsForwardAndAllGetErrors'
                [Qp1,hp1,idp1] = kwta_NN_forward_new(stp1,shunt,Wih,biasih,Who,biasho);
            case 'allHiddenUnitsForwardAndAllGetErrorsShunt1'
                [Qp1,hp1,idp1] = kwta_NN_forward_new(stp1,shunt,Wih,biasih,Who,biasho);
            case 'regularBPNN',
                [Qp1,hp1] = regularBPNN_forward(stp1, Wih,biasih, Who,biasho);
            case 'linearNN'
                 Qp1  = SimpleNN_forward(stp1,Wio,biasio);
            case 'Qtable'
                 Qp1 = Qtable(stp1i,:);
        end
        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
    
        if( ~agentReached2Goal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
           
            % Update Neural Net
           switch functionApproximator,
               case 'kwtaNN',
                   [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'allHiddenUnitsForwardButNoErrorForLosers'
                   [Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'LosersForwardZeroButErrorForAll'
                   [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'allHiddenUnitsForwardAndAllGetErrors'
                   [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'allHiddenUnitsForwardAndAllGetErrorsShunt1'
                   [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'regularBPNN',
                   [Wih,biasih,Who,biasho] = Update_regularBPNN(st,act,h,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'linearNN'
                   [Wio,biasio] = UpdateSimpleNN(st,act,alpha,delta,Wio,biasio);
               case 'Qtable'
                    Qtable(sti,act) = Qtable(sti,act) + alphaTable * delta; 
           end
        else
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            fprintf('Reaching to Goal at episode =%d at step = %d and mean(delta) = %f \n',ei,ts,mean(deltaForStepsOfEpisode));
            delta = rew - Q(act);
            switch functionApproximator,
               case 'kwtaNN',
                   [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'allHiddenUnitsForwardButNoErrorForLosers'
                   [Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'LosersForwardZeroButErrorForAll'
                   [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'allHiddenUnitsForwardAndAllGetErrors'
                   [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'allHiddenUnitsForwardAndAllGetErrorsShunt1'
                   [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'regularBPNN',
                   [Wih,biasih,Who,biasho] = Update_regularBPNN(st,act,h,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'linearNN'
                   [Wio,biasio] = UpdateSimpleNN(st,act,alpha,delta,Wio,biasio);
               case 'Qtable'
                    Qtable(sti,act) = Qtable(sti,act) + alphaTable * delta; 
           end
           break; 
        end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1;
        Q = Qp1;    
        if strcmp(functionApproximator,'Qtable'), sti = stp1i; end
        if strcmp(functionApproximator,'regularBPNN'), h = hp1; end
        if ( strcmp(functionApproximator,'kwtaNN') ...
            || strcmp(functionApproximator,'allHiddenUnitsForwardButNoErrorForLosers') ...
            || strcmp(functionApproximator,'LosersForwardZeroButErrorForAll')...
            || strcmp(functionApproximator,'allHiddenUnitsForwardAndAllGetErrors') ...
            || strcmp(functionApproximator,'allHiddenUnitsForwardAndAllGetErrorsShunt1')),
            id = idp1; h = hp1; 
        end
        ts = ts + 1;
    end % while loop
    meanDeltaForEpisode(ei) = mean(deltaForStepsOfEpisode);
    varianceDeltaForEpisode(ei) =var(deltaForStepsOfEpisode);
    stdDeltaForEpisode(ei) = std(deltaForStepsOfEpisode);
    
    
    if ( ei>500 && abs(meanDeltaForEpisode(ei))< 0.2 && agentReached2Goal ),
            %&& abs(meanDeltaForEpisode(ei))<abs(meanDeltaForEpisode(ei-1) ) ),
        epsilon = bound(epsilon * 0.999,[0.001,0.1]);
    else
        epsilon = bound(epsilon * 1.01,[0.001,0.1]);
    end
    
    if ( abs(meanDeltaForEpisode(ei))<0.1 ) && agentReached2Goal,
        nGoodEpisodes = nGoodEpisodes + 1;
    else
        nGoodEpisodes = 0;
    end
    
    if  abs(mean(deltaForStepsOfEpisode))<0.05 && nGoodEpisodes> nStates*nTilex*nTiley,
        convergence = true;
        fprintf('Convergence at episode: %d \n',ei);
    end
    
    
%     plot(meanDeltaForEpisode)      
%     title(['Episode: ',int2str(ei),' epsilon: ',num2str(epsilon)])    
%     drawnow
    
    ei = ei + 1;

end  % end episode loop

if convergence,
    for i=1:length(xVector)
            for j=1:length(yVector)
               xTest = xVector(i);
               yTest = yVector(j);
               xTestInput = sigmax * sqrt(2*pi) * normpdf(xInputInterval,xTest,sigmax);
               yTestInput = sigmay * sqrt(2*pi) * normpdf(yInputInterval,yTest,sigmay);
               stTest = [xTestInput,yTestInput];
               switch functionApproximator,
                    case 'kwtaNN'
                        [~,~,id] = kwta_NN_forward_new(stTest,shunt,Wih,biasih,Who,biasho);
                    case 'kwtaNN-new'
                        [~,~,id] = kwta_NN_forward_new(stTest,shunt,Wih,biasih,Who,biasho);
                    case 'allHiddenUnitsForwardButNoErrorForLosers'
                        [~,~,id] = kwta_NN_forward_new(stTest,shunt,Wih,biasih,Who,biasho);
                    case 'LosersForwardZeroButErrorForAll'
                        [~,~,id] = kwta_NN_forward(stTest,shunt,Wih,biasih,Who,biasho);
                    case 'allHiddenUnitsForwardAndAllGetErrors'
                        [~,~,id] = kwta_NN_forward_new(stTest,shunt,Wih,biasih,Who,biasho);
                    case  'allHiddenUnitsForwardAndAllGetErrorsShunt1'
                        [~,~,id] = kwta_NN_forward_new(stTest,shunt,Wih,biasih,Who,biasho);
               end
               ultimateWinners(i,j,:) = id; 
            end
    end
end

%% Save Variables
data.meanDeltaForEpisode = meanDeltaForEpisode;
data.varianceDeltaForEpisode = varianceDeltaForEpisode;
data.stdDeltaForEpisode = stdDeltaForEpisode;
data.initialWinners = initialWinners;
data.ultimateWinners = ultimateWinners;

weights.Wih = Wih;
weights.biasih = biasih;
weights.Who = Who;
weights.biasho = biasho;
weights.biasio = biasio;
weights.Wio = Wio;
weights.Qtable = Qtable;

