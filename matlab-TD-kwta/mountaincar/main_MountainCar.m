function [convergence,failure,weights,data] = main_MountainCar(functionApproximator,maxNumEpisodes,nMeshp,nMeshv)
% main_MountainCar.m Performs on-policy sarsa iterative action value funtion 
% estimation for the Mountain Car problem. 
%
% Written during :
% Jacob Rafati    
% Jacob Rafati    start of project: 07/01/2014, San Francisco
% email: jrafatiheravi@ucmerced.edu
% bounds on the position and velocity: 
positionBound = [ -1.5 , +0.6  ]; 
velocityBound = [ -0.07, +0.07 ]; 

% Goal Position
goalPosition = 0.5;

% A matrix of states 
% [statesList,positionVector,velocityVector] = buildStateList(pBound,vBound,nMeshp,nMeshv,goalPosition);
% Compute the mesh (discrete grid) widths: 
dp = diff( positionBound ) / nMeshp;
dv = diff( velocityBound ) / nMeshv; 

% Discritized position and velocity 
% inputPVector = positionBound(1) : dp : positionBound(2);
inputPVector = positionBound(1) : dp : positionBound(2);
inputVVector = velocityBound(1) : dv : velocityBound(2);
nStates = length(inputPVector)*length(inputVVector);
% number of states
statesList = zeros(nStates,2);
index = 1;
for i = 1:length(inputPVector)
    for j = 1:length(inputVVector)
        statesList(index,:) = [inputPVector(i),inputVVector(j)];
        index = index + 1;
    end
end

actionsList = [-1, 0, +1]; % [backward, neutral, forward]
nActions = length(actionsList); % number of actions 

%% kwta and regular BP Neural Network
% Weights from input (x,y) to hidden layer
InputSize = length(inputPVector) + length(inputVVector);
nCellHidden = round(0.7 * nStates);
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
data = struct;

meanDeltaForEpisode = [];
varianceDeltaForEpisode = [];
stdDeltaForEpisode = [];

%% RUN SARSA algorithm
sigmap = dp;
sigmav = dv;

alphaMin = 0.0002;
alphaMax = 0.001;
alpha = alphaMax;

alphaTableMax = 0.2;
alphaTable = alphaTableMax;
alphaTableMin = 0.2;

gamma = 0.99;    % discounted task 

epsilonMax = 0.1; 
epsilon = epsilonMax;  % epsilon greedy parameter
epsilonMin = 0.0001;

% Max number of iteration in ach episde to break the loop if AGENT
% can't reach the GOAL 
maxIteratonEpisode = 1000 * nMeshp / 20;
deltaBound = 0.05;
nGoodEpisodes = 0;
nFailedEpisode = 0;
failure = false;
convergence = false;
actionDistribution = [];
%% Different Max number of episodes
steps = [];

%% test data
meanDeltaForEpisodeTest=[];
stepsTest = [];
iTest = 1;

%% Episode Loops
for ei=1:maxNumEpisodes, % ei<maxNumEpisodes && % ei is counter for episodes
     % initialize the starting state - Continuous state
%     s = [-0.5,0.0];
     s = initializeState(inputPVector,inputVVector);
     p = s(1); v = s(2);
     deltaForStepsOfEpisode = [];
     actionDistributionInEpisode = zeros(1,3);
     if strcmp(functionApproximator,'Qtable'),
         [~,sti] = min(dist(statesList,s'));
     end
     % Gaussian Distribution on continuous state
     pt = sigmap * sqrt(2*pi) * normpdf(inputPVector,p,sigmap);
     vt = sigmav * sqrt(2*pi) * normpdf(inputVVector,v,sigmav);
     % Using st as distributed input for function approximator
     st = [pt,vt];     
     
     % initializing time
     ts = 0;
     switch functionApproximator,
         case 'kwtaNN'
             [Q,h,id] = kwta_NN_forward(st,Wih,biasih,Who,biasho);
         case 'regularBPNN',
             [Q,h] = regularBPNN_forward(st,Wih,biasih, Who,biasho);
         case 'linearNN'
              Q  = SimpleNN_forward(st,Wio,biasio);
         case 'Qtable'
              Q = Qtable(sti,:);
     end
     
     act = e_greedy_policy(Q,nActions,epsilon);
     actionDistributionInEpisode(act) = actionDistributionInEpisode(act) + 1;
 %% Episode While Loop
    while( p < goalPosition && ts<maxIteratonEpisode ),
        % convert the index of the action into an action value
        action = actionsList(act); % action = [-1,0,+1]
        % update state to state+1
        sp1 = MountainCar_updateState(s,action,positionBound,velocityBound);
        p1 = sp1(1); v1 = sp1(2);
        
        switch functionApproximator,
            case 'Qtable',
               [~,stp1i] = min(dist(statesList,sp1'));
        end
        
        ptp1 = sigmap * sqrt(2*pi) * normpdf(inputPVector,p1,sigmap);
        vtp1 = sigmav * sqrt(2*pi) * normpdf(inputVVector,v1,sigmav);
        stp1=[ptp1,vtp1];
        
        % reward/punishment from Environment
        [rew,agentReachedGoal] = getReward(p1,goalPosition);

        
        switch functionApproximator,
            case 'kwtaNN',
                [Qp1,hp1,idp1] = kwta_NN_forward(stp1,Wih,biasih,Who,biasho);
            case 'regularBPNN',
                [Qp1,hp1] = regularBPNN_forward(stp1, Wih,biasih, Who,biasho);
            case 'linearNN'
                 Qp1  = SimpleNN_forward(stp1,Wio,biasio);
            case 'Qtable'
                 Qp1 = Qtable(stp1i,:);
        end
        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
        actionDistributionInEpisode(actp1) = actionDistributionInEpisode(actp1) + 1;
        if( ~agentReachedGoal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
        else
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            % fprintf('Reaching to Goal at episode =%d at step = %d \n',ei,ts);
            delta = rew - Q(act);
        end
        deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
           
            % Update Neural Net
           switch functionApproximator,
               case 'kwtaNN',
                   [Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho); 
               case 'regularBPNN',
                   [Wih,biasih,Who,biasho] = Update_regularBPNN(st,act,h,alpha,delta,Wih,biasih,Who,biasho);
               case 'linearNN'
                   [Wio,biasio] = UpdateSimpleNN(st,act,alpha,delta,Wio,biasio);
               case 'Qtable'
                    Qtable(sti,act) = Qtable(sti,act) + alphaTable * delta; 
           end
           if agentReachedGoal, break; end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1;
        Q = Qp1;    
        if strcmp(functionApproximator,'Qtable'), sti = stp1i; end
        if strcmp(functionApproximator,'regularBPNN'), h = hp1; end
        if strcmp(functionApproximator,'kwtaNN'), id = idp1; h = hp1; end
        ts = ts + 1;
    end % while loop
    
    %% Performance variables, saving as data
    steps(ei) = ts;
    meanDeltaForEpisode(ei) = mean(deltaForStepsOfEpisode);
    varianceDeltaForEpisode(ei) =var(deltaForStepsOfEpisode);
    stdDeltaForEpisode(ei) = std(deltaForStepsOfEpisode);
    actionDistribution(:,ei) = actionDistributionInEpisode./sum(actionDistributionInEpisode);
    %% monitor test points
    if mod(ei,10) == 0,
        % run test
        %%%%%%%%%%%%%%%% TEST BLOCK **********************
     deltaForStepsOfEpisodeTest = [];
     s = [-0.5,0.0];
     p = s(1); v = s(2);
     deltaForStepsOfEpisode = [];
     actionDistributionInEpisode = zeros(1,3);
     if strcmp(functionApproximator,'Qtable'),
         [~,sti] = min(dist(statesList,s'));
     end
     % Gaussian Distribution on continuous state
     pt = sigmap * sqrt(2*pi) * normpdf(inputPVector,p,sigmap);
     vt = sigmav * sqrt(2*pi) * normpdf(inputVVector,v,sigmav);
     % Using st as distributed input for function approximator
     st = [pt,vt];     
     
     % initializing time
     ts = 0;
     switch functionApproximator,
         case 'kwtaNN'
             [Q,h,id] = kwta_NN_forward(st,Wih,biasih,Who,biasho);
         case 'regularBPNN',
             [Q,h] = regularBPNN_forward(st,Wih,biasih, Who,biasho);
         case 'linearNN'
              Q  = SimpleNN_forward(st,Wio,biasio);
         case 'Qtable'
              Q = Qtable(sti,:);
     end
     
     act = e_greedy_policy(Q,nActions,epsilon);
     actionDistributionInEpisode(act) = actionDistributionInEpisode(act) + 1;
        
      while( p < goalPosition && ts<maxIteratonEpisode ),
        % convert the index of the action into an action value
        action = actionsList(act); % action = [-1,0,+1]
        % update state to state+1
        sp1 = MountainCar_updateState(s,action,positionBound,velocityBound);
        p1 = sp1(1); v1 = sp1(2);
        
        switch functionApproximator,
            case 'Qtable',
               [~,stp1i] = min(dist(statesList,sp1'));
        end
        
        ptp1 = sigmap * sqrt(2*pi) * normpdf(inputPVector,p1,sigmap);
        vtp1 = sigmav * sqrt(2*pi) * normpdf(inputVVector,v1,sigmav);
        stp1=[ptp1,vtp1];
        
        % reward/punishment from Environment
        [rew,agentReachedGoal] = getReward(p1,goalPosition);

        
        switch functionApproximator,
            case 'kwtaNN',
                [Qp1,hp1,idp1] = kwta_NN_forward(stp1,Wih,biasih,Who,biasho);
            case 'regularBPNN',
                [Qp1,hp1] = regularBPNN_forward(stp1, Wih,biasih, Who,biasho);
            case 'linearNN'
                 Qp1  = SimpleNN_forward(stp1,Wio,biasio);
            case 'Qtable'
                 Qp1 = Qtable(stp1i,:);
        end
        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
        actionDistributionInEpisode(actp1) = actionDistributionInEpisode(actp1) + 1;
        if( ~agentReachedGoal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
        else
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            % fprintf('Reaching to Goal at episode =%d at step = %d \n',ei,ts);
            delta = rew - Q(act);
        end
        deltaForStepsOfEpisodeTest = [deltaForStepsOfEpisodeTest,delta];
           
        if agentReachedGoal, break; end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1;
        Q = Qp1;    
        if strcmp(functionApproximator,'Qtable'), sti = stp1i; end
        if strcmp(functionApproximator,'regularBPNN'), h = hp1; end
        if strcmp(functionApproximator,'kwtaNN'), id = idp1; h = hp1; end
        ts = ts + 1;
      end % while loop
      stepsTest(iTest) = ts;
      meanDeltaForEpisodeTest(iTest) = mean(deltaForStepsOfEpisodeTest);
      iTest = iTest + 1;
    end
    %%%%%%%%%%%%%%%% END TEST BLOCK ******************

    %% Exploration vs. Exploitation    
    if agentReachedGoal && (ei > 50),
        alpha = bound(alpha * 0.99,[alphaMin,alphaMax]);
        alphaTable = bound(alphaTable * 0.99,[alphaTableMin,alphaTableMax]);
        epsilon = bound(epsilon * 0.999,[epsilonMin,epsilonMax]);
    else
        epsilon = bound(epsilon * 1.01,[epsilonMin,epsilonMax]);
        alpha = bound(alpha * 1.1,[alphaMin,alphaMax]);
        alphaTable = bound(alphaTable * 1.1,[alphaTableMin,alphaTableMax]);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% failure conditions
%     if ~agentReachedGoal && abs( meanDeltaForEpisode(ei) ) < deltaBound
%         nFailedEpisode = nFailedEpisode + 1;
%     else
%         nFailedEpisode = 0;
%     end
%     
%     if nFailedEpisode == 200,
%         failure = true;
%         break;
%     end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% performance plots, comment on runs  
%     subplot(2,1,1);    
%     plot(meanDeltaForEpisode)      
%     title(['Episode: ',int2str(ei),' epsilon: ',num2str(epsilon)])    
%     drawnow;
%     subplot(2,1,2); 
%     plot(steps)
%     drawnow;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Convergence conditions
    if agentReachedGoal && (ei > 100) && ...
            abs( meanDeltaForEpisode(ei) ) < deltaBound,
        nGoodEpisodes = nGoodEpisodes + 1;
    else
        nGoodEpisodes = 0;
    end
     
    if nGoodEpisodes == 500, 
        convergence = true;
        break;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end  % end episode loop

%% Save Variables
data.meanDeltaForEpisode = meanDeltaForEpisode;
data.varianceDeltaForEpisode = varianceDeltaForEpisode;
data.stdDeltaForEpisode = stdDeltaForEpisode;
data.actionDistribution = actionDistribution;
data.meanDeltaForEpisodeTest = meanDeltaForEpisodeTest;
data.stepsTest = stepsTest;
data.steps = steps;
weights.Wih = Wih;
weights.biasih = biasih;
weights.Who = Who;
weights.biasho = biasho;
weights.biasio = biasio;
weights.Wio = Wio;
weights.Qtable = Qtable;




