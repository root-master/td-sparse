clc, close all, clear all;
withBias = 1;

nMeshx = 20; nMeshy = 20;

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
% Weights from input (s_sq, g_sq) to hidden layer
% s_sq: square representation of s
% g_sq: square representation of g
s_sq_dim = [length(xInputInterval), length(yInputInterval)];
q_sq_dim = [length(xInputInterval), length(yInputInterval)];

n_h_1_dim = [41, 41];

n_o_dim = [4,1];

%%%%%%%%%%%% INIT WEIGHTS %%%%%%%%%%%%
mu = 0.1;
W_s_h_1 = 0;



alpha = 0.003;

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
        save(filename,'Wih','biasih','Who','biasho');
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
     [Q,h,id] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
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
        [Qp1,hp1,idp1] = kwta_NN_forward_new(stp1,shunt,Wih,biasih,Who,biasho);
        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
    
        if( ~agentReached2Goal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
           
            % Update Neural Net
            [Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho);
        else
            delta = rew - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
            [Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho);
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            fprintf('Success: episode = %d, s0 = (%g , %g), goal: (%g , %g), step = %d, mean(delta) = %f \n',ei,s0,g,ts,mean(deltaForStepsOfEpisode));
            break; 
        end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1; id = idp1; h = hp1; 
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