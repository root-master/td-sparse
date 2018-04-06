clc, close all, clear all;
withBias = 1;

nMeshx = 20; nMeshy = 20;
nTilex = 1; nTiley = 1;

functionApproximator = 'kwtaNN';
shunt = 1.0;

% control task could be 'grid_world' or 'puddle_world'
task = 'puddle_world';
% function approximator can be either 'kwtaNN' or 'regularBPNN'


% goal in continouos state
% g is goal and it is dynamic this time

% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;

% the number of states -- This is the gross mesh states ; the 1st tiling 
nStates = ( length(xInputInterval) * length(yInputInterval) ); 

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

%% kwta and regular BP Neural Network
% Weights from input (x,y,x_goal,y_goal) to hidden layer
InputSize =  2 * ( length(xInputInterval) + length(yInputInterval ));
nCellHidden1 = 2 * nStates;
nCellHidden2 = round(0.5 * nStates);

mu = 0.001;
Wih = mu * (rand(InputSize,nCellHidden1) - 0.5);
biasih = mu * ( rand(1,nCellHidden1) - 0.5 );

Wh1h2 = mu * (rand(nCellHidden1,nCellHidden2) - 0.5);
biash1h2 = mu * ( rand(1,nCellHidden2) - 0.5 );


% Weights from hidden layer to output
Who = mu * (rand(nCellHidden2,nActions) - 0.5);
biasho = mu * ( rand(1,nActions) - 0.5 );

alpha = 0.0005;

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

gamma = 0.99;    % discounted task 
epsilon = 0.1;  % epsilon greedy parameter
epsilon_max = 0.1;

% Max number of iteration in ach episde to break the loop if AGENT
% can't reach the GOAL 
maxIteratonEpisode = 4 * (nMeshx * nTilex + nMeshy * nTiley);
             
%% Input of function approximator
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
maxNumEpisodes = 1000000;

nGoodEpisodes = 0; % a variable for checking the convergence
convergence = false;
agentReached2Goal = false;
agentBumped2wall = false;
%% Episode Loops
ei = 0;
delta_sum = [];
save_episodes = 0:1000:1000000;
total_num_steps = 0;
g = [1,1]; % just initalization --> it's gonna change
while (ei < maxNumEpisodes && ~convergence ), % ei<maxNumEpisodes && % ei is counter for episodes
    if ismember(ei,save_episodes)
        filename = ['./results/weights',int2str(ei),'.mat'];
        %save(filename,'Wih','biasih','Who','biasho');
    end
    ei = ei + 1;
    
    deltaForStepsOfEpisode = [];
     % initialize the starting state - Continuous state
     s = initializeState(xVector,yVector);
     s0 = s;
     goalinPuddle = true;
     while (goalinPuddle),
        g = initializeState(xVector,yVector);
        [goalinPuddle,~] = CreatePuddle(g);
     end
     % Gaussian Distribution on continuous state
     sx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,s(1),sigmax);
     sy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,s(2),sigmay);
     gx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,g(1),sigmax);
     gy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,g(2),sigmay);
     % Using st as distributed input for function approximator
     st = [sx,sy,gx,gy];     
          
     % initializing time
     ts = 1;
     [Q,h_1,h_1_id,h_2,h_2_id] = kwta_NN_forward_2_layer(st, Wih,biasih, Wh1h2,biash1h2, Who,biasho);
     act = e_greedy_policy(Q,nActions,epsilon);

    %% Episode While Loop
    while( ~(s(1)==g(1) && s(2)==g(2)) && ts < maxIteratonEpisode),
        % update state to state+1
        sp1 = UPDATE_STATE(s,act,xgrid,xVector,ygrid,yVector);
        xp1 = sp1(1); yp1 = sp1(2);
        % PDF of stp1
        sxp1 = sigmax * sqrt(2*pi) * normpdf(xInputInterval,xp1,sigmax);
        syp1 = sigmay * sqrt(2*pi) * normpdf(yInputInterval,yp1,sigmay);
        stp1=[sxp1,syp1,gx,gy];
        
        if ( sp1(1)==g(1) && sp1(2)==g(2) ),
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
        [Qp1,hp1_1,hp1_1_id,hp1_2,hp1_2_id] = kwta_NN_forward_2_layer(st, Wih,biasih, Wh1h2,biash1h2, Who,biasho);
        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
    
        if( ~agentReached2Goal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
           
            % Update Neural Net
            [ Wih,biasih,Wh1h2,biash1h2,Who,biasho] = Update_kwtaNN_2_layer(st,act,h_1,h_1_id,h_2,h_2_id,alpha,delta,Wih,biasih, Wh1h2,biash1h2, Who,biasho);
        else
            delta = rew - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
            [ Wih,biasih,Wh1h2,biash1h2,Who,biasho] = Update_kwtaNN_2_layer(st,act,h_1,h_1_id,h_2,h_2_id,alpha,delta,Wih,biasih, Wh1h2,biash1h2, Who,biasho);            
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            fprintf('Success: episode = %d, s0 = (%g , %g), goal: (%g , %g), step = %d, mean(delta) = %f \n',ei,s0,g,ts,mean(deltaForStepsOfEpisode));
            break; 
        end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1; h_1_id = hp1_1_id; h_1 = hp1_1;
        h_2 = hp1_2; h_2_id = hp1_2_id;
        Q = Qp1;    
        ts = ts + 1;
    end % while loop
    %epsilon = epsilon_max/ei;
    total_num_steps = total_num_steps + ts;
    meanDeltaForEpisode(ei) = mean(deltaForStepsOfEpisode);
    delta_sum(ei) = sum(deltaForStepsOfEpisode);
    varianceDeltaForEpisode(ei) =var(deltaForStepsOfEpisode);
    stdDeltaForEpisode(ei) = std(deltaForStepsOfEpisode);
    
    
    if ( ei>1000) && (abs(sum(delta_sum)) / total_num_steps) < 0.2 && agentReached2Goal,
            %&& abs(meanDeltaForEpisode(ei))<abs(meanDeltaForEpisode(ei-1) ) ),
        epsilon = bound(epsilon * 0.99,[0.001,0.1]);
    else
        epsilon = bound(epsilon * 1.01,[0.001,0.1]);
    end
    
    if abs(sum(delta_sum) ) / total_num_steps< 0.1 && agentReached2Goal,
        nGoodEpisodes = nGoodEpisodes + 1;
    else
        nGoodEpisodes = 0;
    end
    
    if  abs(sum(delta_sum) ) / total_num_steps< 0.05 && nGoodEpisodes> nStates*nTilex*nTiley,
        convergence = true;
        fprintf('Convergence at episode: %d \n',ei);
    end
    
    
%     plot(meanDeltaForEpisode)      
%     title(['Episode: ',int2str(ei),' epsilon: ',num2str(epsilon)])    
%     drawnow
    

end  % end episode loop