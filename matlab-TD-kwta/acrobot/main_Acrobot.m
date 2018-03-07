function [convergence,failure,weights,data,nMesh,functionApproximator] = main_Acrobot(functionApproximator,maxNumEpisodes,nMesh)
% main_MountainCar.m Performs on-policy sarsa iterative action value funtion 
% estimation for the Mountain Car problem. 
%
% Written during :
% Jacob Rafati    
% Jacob Rafati    start of project: 07/01/2014, San Francisco
% email: jrafatiheravi@ucmerced.edu



grafica = false;

% state variables domain:
% [angle positions and velocities (theta1,theta2,omega1,omega2)]: 
theta1_bound = [-pi , pi]; 
theta2_bound = [-pi , pi];
omega1_bound = [-4*pi , 4*pi]; 
omega2_bound = [-9*pi,9*pi];

% number of mesh for descritzing the state space
nMesh_theta1 = nMesh(1);
nMesh_omega1 = nMesh(2);
nMesh_theta2 = nMesh(3);
nMesh_omega2 = nMesh(4);

dtheta1 = diff( theta1_bound ) / nMesh_theta1;
dtheta2 = diff( theta2_bound ) / nMesh_theta2;
domega1 = diff( omega1_bound ) / nMesh_omega1;
domega2 = diff( omega2_bound ) / nMesh_omega2;

% input vector corressoponding to state variables 
theta1_vec = theta1_bound(1) : dtheta1 : theta1_bound(2);
theta2_vec = theta2_bound(1) : dtheta2 : theta2_bound(2);
omega1_vec = omega1_bound(1) : domega1 : omega1_bound(2);
omega2_vec = omega2_bound(1) : domega2 : omega2_bound(2);

% number of states
nStates = length(theta1_vec) * length(theta2_vec) * ...
    length(omega1_vec) * length(omega2_vec);
statesList = zeros(nStates,4);
index = 1;
for i = 1:length(theta1_vec)
    for j = 1:length(omega1_vec)
        for k = 1:length(theta2_vec)
            for l = 1:length(omega2_vec)
                statesList(index,:) = ...
                    [theta1_vec(i),omega1_vec(j),theta2_vec(k),omega2_vec(l)];
                    index = index + 1;
            end
        end
    end
end

%stateList_reshaped = reshape(statesList, [])

actionsList = [-1, 0, 1]; % apply torque to [backward, neutral?, forward]
nActions = length(actionsList); % number of actions 

%% kwta and regular BP Neural Network
inputSize = length(theta1_vec) + length(theta2_vec) + ...
    length(omega1_vec) + length(omega2_vec);
nCellHidden = 200 * inputSize;
% nCxellHidden = round(0.1 * nStates);
% noise amplitude
mu = 0.1;
% Weights from input (x,y) to hidden layer
Wih = mu * (rand(inputSize,nCellHidden) - 0.5);
biasih = mu * ( rand(1,nCellHidden) - 0.5 );
% Weights from hidden layer to output
Who = mu * (rand(nCellHidden,nActions) - 0.5);
biasho = mu * ( rand(1,nActions) - 0.5 );

%% Linear Neural Net
mu = 0.1; % amplitude of random weights
Wio = mu * (rand(inputSize,nActions) - 0.5);
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

sigma_theta1 = dtheta1;
sigma_theta2 = dtheta2;
sigma_omega1 = domega1;
sigma_omega2 = domega2;

alphaMin = 0.00001;
alphaMax = 0.0001;
alpha = alphaMax;

alphaTableMax = 0.5;
alphaTable = alphaTableMax;
alphaTableMin = 0.5;

gamma = 0.99;    % discounted task 

epsilonMax = 0.05; 
epsilon = epsilonMax;  % epsilon greedy parameter
epsilonMin = 0.0001;

% Max number of iteration in ach episde to break the loop if AGENT
                                             % - can't reach the GOAL 
maxIteratonEpisode = 2000;
deltaBound = 0.05;
nGoodEpisodes = 0;
failure = false;
convergence = false;
%% Different Max number of episodes
steps = [];

%% test data
meanDeltaForEpisodeTest=[];
stepsTest = [];
indexTest = 1;

%% Episode Loops

for ei = 1:maxNumEpisodes, 
     % initialize the starting state - Continuous state
     s = [0.0,0.0,0.0,0.0];
     % s = initializeState(theta1_vec,omega1_vec,theta2_vec,omega2_vec);
     theta1 = s(1); omega1 = s(2); 
     theta2 = s(3); omega2 = s(4); 
     deltaForStepsOfEpisode = [];
     if strcmp(functionApproximator,'Qtable'),
         % find the closest state in statesList to s
         sti = dsearchn(statesList,s);
         %[~,sti] = min(dist(statesList,s'));
     end
     % Gaussian Distribution on continuous state
     theta1_input = sigma_theta1 * sqrt(2*pi) * normpdf(theta1_vec,theta1,sigma_theta1);
     omega1_input = sigma_omega1 * sqrt(2*pi) * normpdf(omega1_vec,omega1,sigma_omega1);
     theta2_input = sigma_theta2 * sqrt(2*pi) * normpdf(theta2_vec,theta2,sigma_theta2);
     omega2_input = sigma_omega2 * sqrt(2*pi) * normpdf(omega2_vec,omega2,sigma_omega2);

     % Using st as distributed input for function approximator
     st = [theta1_input,omega1_input,theta2_input,omega2_input];     
     
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
     agentReachedGoal = false;

 %% Episode While Loop
    while( ~agentReachedGoal && ts<maxIteratonEpisode ),
        % convert the index of the action into an action value
        action = actionsList(act); % actionList = [-1,0,+1]
        % update state to state+1
        [sp1,agentReachedGoal] = updateState(s,action,omega1_bound,omega2_bound);
        % reward/punishment from Environment
        [rew] = getReward(agentReachedGoal);
        
        theta1_p1 = sp1(1); omega1_p1 = sp1(2);
        theta2_p1 = sp1(3); omega2_p1 = sp1(4); 
        
        if strcmp(functionApproximator,'Qtable'),
               % [~,stp1i] = min(dist(statesList,sp1'));
               stp1i = dsearchn(statesList,sp1);
        end
        
        theta1_p1_input = sigma_theta1 * sqrt(2*pi) * normpdf(theta1_vec,theta1_p1,sigma_theta1);
        omega1_p1_input = sigma_omega1 * sqrt(2*pi) * normpdf(omega1_vec,omega1_p1,sigma_omega1);
        theta2_p1_input = sigma_theta2 * sqrt(2*pi) * normpdf(theta2_vec,theta2_p1,sigma_theta2);
        omega2_p1_input = sigma_omega2 * sqrt(2*pi) * normpdf(omega2_vec,omega2_p1,sigma_omega2);

        stp1=[theta1_p1_input,omega1_p1_input,theta2_p1_input,omega2_p1_input];
        
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
        
        if( ~agentReachedGoal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
        else
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            fprintf('Reaching to Goal at episode =%d at step = %d \n',ei,ts);
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
    
    steps(ei) = ts;
    meanDeltaForEpisode(ei) = mean(deltaForStepsOfEpisode);
    varianceDeltaForEpisode(ei) =var(deltaForStepsOfEpisode);
    stdDeltaForEpisode(ei) = std(deltaForStepsOfEpisode);
    

%% Exploration vs. Exploitation    
    if agentReachedGoal && (ei > 50),
        alpha = bound(alpha * 0.99,[alphaMin,alphaMax]);
        alphaTable = bound(alphaTable * 0.99,[alphaTableMin,alphaTableMax]);
        epsilon = bound(epsilon * 0.99,[epsilonMin,epsilonMax]);
    else
        epsilon = bound(epsilon * 1.01,[epsilonMin,epsilonMax]);
        alpha = bound(alpha * 1.01,[alphaMin,alphaMax]);
        alphaTable = bound(alphaTable * 1.01,[alphaTableMin,alphaTableMax]);
    end
    
    
%     subplot(2,1,1);    
%     plot(meanDeltaForEpisode)      
%     title(['Episode: ',int2str(ei),' epsilon: ',num2str(epsilon)])    
%     drawnow;
%     subplot(2,1,2); 
%     plot(steps)
%     drawnow;
    
    %}
    %% Convergence conditions
    
    if agentReachedGoal && (ei > 1000) && ...
            abs( meanDeltaForEpisode(ei) ) < deltaBound && ...
            mean( steps(end-99:end) ) < 0.95 * max( steps(end-99:end) ) && ...
            0.95 * min( steps(end-99:end) ) < mean( steps(end-99:end) ),
        nGoodEpisodes = nGoodEpisodes + 1;
    else
        nGoodEpisodes = 0;
    end
     
    if nGoodEpisodes == 100, 
        convergence = true;
        break;
    end
    %}
    
    if (mod(ei,100)==0),
        weights.Wih = Wih;
        weights.biasih = biasih;
        weights.Who = Who;
        weights.biasho = biasho;
        weights.biasio = biasio;
        weights.Wio = Wio;
        weights.Qtable = Qtable;
        [ts,deltaForStepsOfEpisode,~] = runTest(nMesh,functionApproximator,weights,Qtable,statesList,grafica);
        meanDeltaForEpisodeTest(indexTest) = mean(deltaForStepsOfEpisode);
        stepsTest(indexTest) = ts;
        indexTest = indexTest + 1;
    end

end  % end episode loop

%% Save Variables
    data.meanDeltaForEpisode = meanDeltaForEpisode;
    data.varianceDeltaForEpisode = varianceDeltaForEpisode;
    data.stdDeltaForEpisode = stdDeltaForEpisode;
    data.steps = steps;
    data.statesList = statesList;
    weights.Wih = Wih;
    weights.biasih = biasih;
    weights.Who = Who;
    weights.biasho = biasho;
    weights.biasio = biasio;
    weights.Wio = Wio;
    weights.Qtable = Qtable;
    data.stepsTest = stepsTest;
    data.meanDeltaForEpisodeTest = meanDeltaForEpisodeTest;

