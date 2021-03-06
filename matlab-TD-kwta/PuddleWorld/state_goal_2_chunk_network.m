clc, close all, clear all;
withBias = 1;

nMeshx = 10; nMeshy = 10;
nTilex = 1; nTiley = 1;

functionApproximator = 'kwtaNN';
shunt = 1.0;

% control task could be 'grid_world' or 'puddle_world'
task = 'puddle_world';

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
InputSize =  length(xInputInterval) + length(yInputInterval);
nCellHidden1 = round(nStates/2);
nCellHidden2 = round(nStates);

mu = 0.01;

Wsh = mu * (rand(InputSize,nCellHidden1) - 0.5);

Wgh = mu * (rand(InputSize,nCellHidden1) - 0.5);

bsh =  mu * ( rand(1,nCellHidden1) - 0.5 );

bgh =  mu * ( rand(1,nCellHidden1) - 0.5 );

Wsgh = mu * (rand(2*nCellHidden1,nCellHidden2) - 0.5);
bsgh = mu * (rand(1,nCellHidden2) - 0.5);

Who =  mu * (rand(nCellHidden2,nActions) - 0.5);
bho =  mu * (rand(1,nActions) - 0.5);

epsilon_max = 0.1;
epsilon_min = 0.0001;
epsilon = epsilon_max;  % epsilon greedy parameter

alpha_min = 0.00000001;
alpha_max = 0.0001;
alpha = alpha_max;

gamma = 0.99;    % discounted task 

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
radius = 0.15;
while (ei < maxNumEpisodes && ~convergence ), % ei<maxNumEpisodes && % ei is counter for episodes
    if ismember(ei,save_episodes)
        filename = ['./results/weights',int2str(ei),'.mat'];
        %save(filename,'Wih','biasih','Who','biasho');
    end
    
    
    if mod(ei,1000)==0
        [successful_key_door_episodes, successful_key_episodes, successful_easy_episodes, scores_vec, total_episodes] = test_score_success_func_2_hidden(Wsh,Wgh,bsh,bgh,Wsgh,bsgh,Who,bho);        
        fprintf('average success: %.4f \n',length(successful_key_episodes)/total_episodes);
        pause(5)
    end

    
    deltaForStepsOfEpisode = [];
     % initialize the starting state - Continuous state
     s0 = initializeState(xVector,yVector);
     s = s0;
%      goalinPuddle = true;
%      while (goalinPuddle),
%         g = initializeState(xVector,yVector);
%         [goalinPuddle,~] = CreatePuddle(g);
%      end
     g = neighbor_state(s0,xVector,yVector,radius);


% Gaussian Distribution on continuous state
     sx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,s(1),sigmax);
     sy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,s(2),sigmay);
     gx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,g(1),sigmax/2);
     gy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,g(2),sigmay/2);
     % Using st as distributed input for function approximator
     st = [sx,sy];
     gt = [gx,gy];
          
     % initializing time
     ts = 1;
     [Q,sh,gh,h] = kwta_NN_forward_2chunk(st,gt,Wsh,Wgh,bsh,bgh,Wsgh,bsgh,Who,bho);
     act = e_greedy_policy(Q,nActions,epsilon);
     ei = ei + 1;
    %% Episode While Loop
    while( ~(s(1)==g(1) && s(2)==g(2)) && ts < maxIteratonEpisode),
        % update state to state+1
        sp1 = UPDATE_STATE(s,act,xgrid,xVector,ygrid,yVector);
        xp1 = sp1(1); yp1 = sp1(2);
        % PDF of stp1
        sxp1 = sigmax * sqrt(2*pi) * normpdf(xInputInterval,xp1,sigmax);
        syp1 = sigmay * sqrt(2*pi) * normpdf(yInputInterval,yp1,sigmay);
        stp1=[sxp1,syp1];
        
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
        rew = ENV_REWARD(sp1,agentReached2Goal,agentBumped2wall);
        [Qp1,shp1,ghp1,hp1] = kwta_NN_forward_2chunk(stp1,gt,Wsh,Wgh,bsh,bgh,Wsgh,bsgh,Who,bho);
        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
    
        if( ~agentReached2Goal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
           
            % Update Neural Net
            [Wsh,Wgh,bsh,bgh,Wsgh,bsgh, Who,bho] = Update_kwtaNN_2_chunk(st,gt,act,sh,gh,h,alpha,delta,Wsh,Wgh,bsh,bgh,Wsgh,bsgh,Who,bho);        
        else
            delta = rew - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
            [Wsh,Wgh,bsh,bgh,Wsgh,bsgh, Who,bho] = Update_kwtaNN_2_chunk(st,gt,act,sh,gh,h,alpha,delta,Wsh,Wgh,bsh,bgh,Wsgh,bsgh, Who,bho);
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            fprintf('Success: episode = %d, s0 = (%g , %g), goal: (%g , %g), step = %d, mean(delta) = %f \n',ei,s0,g,ts,mean(deltaForStepsOfEpisode));
            break; 
        end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1; sh=shp1; gh=ghp1; h=hp1;
        Q = Qp1;    
        ts = ts + 1;
    end % while loop
    %epsilon = epsilon_max/ei;
    total_num_steps = total_num_steps + ts;
    meanDeltaForEpisode(ei) = mean(deltaForStepsOfEpisode);
    delta_sum(ei) = sum(deltaForStepsOfEpisode);
    varianceDeltaForEpisode(ei) =var(deltaForStepsOfEpisode);
    stdDeltaForEpisode(ei) = std(deltaForStepsOfEpisode);
    
%     if ( ei>1000) && (abs(sum(delta_sum)) / total_num_steps) < 0.2 && agentReached2Goal,
%             %&& abs(meanDeltaForEpisode(ei))<abs(meanDeltaForEpisode(ei-1) ) ),
%         epsilon = bound(epsilon * 0.99,[0.001,epsilon_max]);
%     else
%         epsilon = bound(epsilon * 1.01,[0.001,epsilon_max]);
%     end
%     
%     if abs(sum(delta_sum) ) / total_num_steps< 0.1 && agentReached2Goal,
%         nGoodEpisodes = nGoodEpisodes + 1;
%     else
%         nGoodEpisodes = 0;
%     end
    
    if mod(ei,1000)==0    
        %alpha = bound(alpha * 0.99,[alpha_min,alpha_max]);
        epsilon = bound(epsilon * 0.99,[epsilon_min,epsilon_max]);
    end
    
    if mod(ei,5000)==0
        radius = bound(1.05 * radius, [0.1,1.0]);
    end

    if  abs(sum(delta_sum) ) / total_num_steps< 0.05 && nGoodEpisodes> nStates*nTilex*nTiley,
        convergence = true;
        fprintf('Convergence at episode: %d \n',ei);
    end
    
    
%     plot(meanDeltaForEpisode)      
%     title(['Episode: ',int2str(ei),' epsilon: ',num2str(epsilon)])    
%     drawnow
    

end  % end episode loop