clc, close all, clear all;
withBias = 1;

nMeshx = 10; nMeshy = 10;
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
InputSize = length(xInputInterval) + length(yInputInterval );
goal_size = nStates;
nCellHidden = round( nStates );
mu = 0.1;

W_ih_big = mu * (rand(InputSize,nCellHidden,goal_size) - 0.5);
b_ih_big = mu * ( rand(1,nCellHidden,goal_size) - 0.5 );

W_ho_big = mu * (rand(nCellHidden,nActions,goal_size) - 0.5);
b_ho_big = mu * ( rand(1,nActions,goal_size) - 0.5 );

% Wih = mu * (rand(InputSize,nCellHidden) - 0.5);
% biasih = mu * ( rand(1,nCellHidden) - 0.5 );
% % Weights from hidden layer to output
% Who = mu * (rand(nCellHidden,nActions) - 0.5);
% biasho = mu * ( rand(1,nActions) - 0.5 );

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

gamma = 0.99;    % discounted task 
epsilon_max = 0.1;
epsilon_min = 0.01;
epsilon = epsilon_max;  % epsilon greedy parameter


alpha_min = 0.002;
alpha_max = 0.005;
alpha = alpha_max;
% Max number of iteration in ach episde to break the loop if AGENT
% can't reach the GOAL 

             
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
maxNumEpisodes = 10000000;

nGoodEpisodes = 0; % a variable for checking the convergence
convergence = false;
agentReached2Goal = false;
agentBumped2wall = false;
%% Episode Loops
ei = 0;
delta_sum = [];
save_episodes = 0:1000:1000000;
total_num_steps = 0;
radius = 0.11;
results_basic_score = [];
results_basic_success = [];
results_hard_score = [];
results_medium_score = [];
results_hard_success = [];
results_medium_success = [];
results_easy_success = [];
results_hard_score_best = [];
results_medium_score_best = [];
results_basic_score_best = [];

while (ei < maxNumEpisodes && ~convergence ), % ei<maxNumEpisodes && % ei is counter for episodes
    if ismember(ei,save_episodes)
        filename = ['./results/weights',int2str(ei),'.mat'];
        %save(filename,'Wih','biasih','Who','biasho');
    end
    
    if mod(ei,1000)==0
        [successful_key_door_episodes, successful_key_episodes, successful_easy_episodes, successful_basic_key_door, scores_vec,scores_vec_basic,score_key, total_episodes] = test_func_seperate_policy(W_ih_big, b_ih_big, W_ho_big, b_ho_big);        
        results_basic_score = [results_basic_score,mean(scores_vec_basic)];
        results_basic_success = [results_basic_success,length(successful_basic_key_door)/total_episodes];
        results_hard_score = [results_hard_score, mean(scores_vec)];
        results_hard_success = [results_hard_success,length(successful_key_door_episodes)/total_episodes];
        results_medium_success = [results_medium_success, length(successful_key_episodes)/total_episodes];
        results_medium_score = [results_medium_score,mean(score_key)];
        results_easy_success = [results_easy_success, length(successful_easy_episodes)/total_episodes];
        results_hard_score_best = [results_hard_score_best,max(scores_vec)];
        results_medium_score_best = [results_medium_score_best,max(score_key)];
        results_basic_score_best = [results_basic_score_best,max(scores_vec_basic)];
        
        fprintf('average success hard     : %.4f \n',length(successful_key_door_episodes)/total_episodes);
        fprintf('average success basic    : %.4f \n',length(successful_basic_key_door)/total_episodes);
        fprintf('average success medium   : %.4f \n',length(successful_key_episodes)/total_episodes);
        fprintf('average success easy     : %.4f \n',length(successful_easy_episodes)/total_episodes);
        fprintf('average score   hard     : %.4f \n',mean(scores_vec));
        fprintf('average score   basic    : %.4f \n',mean(scores_vec_basic));
        fprintf('average score   medium   : %.4f \n',mean(score_key));
        fprintf('max score       hard     : %.4f \n',max(scores_vec));
        fprintf('max score       basic    : %.4f \n',max(scores_vec_basic));
        fprintf('max score       medium   : %.4f \n',max(score_key));

        pause(5)
    end
      
     goalinPuddle = true;
     while (goalinPuddle),
        g = initializeState(xVector,yVector);
        [goalinPuddle,~] = CreatePuddle(g);
     end
     % initialize the starting state - Continuous state
     s0inPuddle = true;
     while (s0inPuddle),
        s0 = initializeState(xVector,yVector);
        [s0inPuddle,~] = CreatePuddle(s0);
     end
     %s0 = initializeState(xVector,yVector);
     s = s0;
     % g = neighbor_state(s0,xVector,yVector,radius);
     
     [~,gidx] = min(dist(g(1),xInputInterval));
     [~,gidy] = min(dist(g(2),yInputInterval));
     gid = sub2ind([length(xVector),length(yVector)],gidx,gidy);
 
     Wih = W_ih_big(:,:,gid);
     biasih = b_ih_big(:,:,gid);
     Who = W_ho_big(:,:,gid);
     biasho = b_ho_big(:,:,gid);
     % Gaussian Distribution on continuous state
     sx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,s(1),sigmax);
     sy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,s(2),sigmay);
     % Using st as distributed input for function approximator
    
     st = [sx,sy];     
          
     % initializing time
     ts = 1;
     [Q,h,id] = kwta_NN_forward_new(st,Wih,biasih,Who,biasho);
     act = e_greedy_policy(Q,nActions,epsilon);
     ei = ei + 1;
     deltaForStepsOfEpisode = [];
     maxIteratonEpisode = 4*nMeshx;
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
        [Qp1,hp1,idp1] = kwta_NN_forward_new(stp1,Wih,biasih,Who,biasho);
        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
    
        if( ~agentReached2Goal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
           
            % Update Neural Net
            [Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho);
             W_ih_big(:,:,gid) = Wih ;
             b_ih_big(:,:,gid) = biasih;
             W_ho_big(:,:,gid) = Who;
             b_ho_big(:,:,gid) = biasho;
        else
            delta = rew - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
            [Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho);
             W_ih_big(:,:,gid) = Wih ;
             b_ih_big(:,:,gid) = biasih;
             W_ho_big(:,:,gid) = Who;
             b_ho_big(:,:,gid) = biasho;

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

    %% Exploration vs. Exploitation    
    if agentReached2Goal,
        %alpha = bound(alpha * 0.999,[alpha_min,alpha_max]);
        epsilon = bound(epsilon * 0.999,[epsilon_min,epsilon_max]);
    else
        epsilon = bound(epsilon * 1.001,[epsilon_min,epsilon_max]);
        %alpha = bound(alpha * 1.001,[alpha_min,alpha_max]);
    end
    
    if mod(ei,10000) == 0
        file_name = 'everything_April_12th.mat';
        save(file_name) 
    end
%     if mod(ei,1000)==0    
%         alpha = bound(alpha * 0.99,[alpha_min,alpha_max]);
%         epsilon = bound(epsilon * 0.99,[epsilon_min,epsilon_max]);
%     end
%     
%     if mod(ei,5000)==0
%         radius = bound(1.05 * radius, [0.1,1.0]);
%     end

    
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
%     
%     if  abs(sum(delta_sum) ) / total_num_steps< 0.05 && nGoodEpisodes> nStates*nTilex*nTiley,
%         convergence = true;
%         fprintf('Convergence at episode: %d \n',ei);
%     end


    
%     plot(meanDeltaForEpisode)      
%     title(['Episode: ',int2str(ei),' epsilon: ',num2str(epsilon)])    
%     drawnow
    

end  % end episode loop


