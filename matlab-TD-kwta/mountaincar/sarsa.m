function [weights,data,convergence] = sarsa(task,functionApproximator,weights,data,nMeshx,nTilex,nMeshy,nTiley)
Wih = weights.Wih;
biasih = weights.biasih;
Who = weights.Who;
biasho = weights.biasho;
Wio = weights.Wio;
biasio = weights.biasio;
Qtable = weights.Qtable;

% initialWinners = data.initialWinners;
% ultimateWinners = data.ultimateWinners;

meanDeltaForEpisode = data.meanDeltaForEpisode;
varianceDeltaForEpisode = data.varianceDeltaForEpisode;
stdDeltaForEpisode = data.stdDeltaForEpisode; 

s_end = 0.45;

alpha = 0.005;
alphaTable = 0.4;

nActions = 3;

gamma = 0.99;    % discounted task 
epsilon = 0.1;  % for our epsilon greedy policy% epsilon will be decrease exponentialy

% Max number of iteration in ach episde to break the loop if AGENT
% can't reach the GOAL 
maxIteratonEpisode = 4 * (nMeshx * nTilex + nMeshy * nTiley);
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
deltaForStepsOfEpisode = [];

nGoodEpisodes = 0; % a variable for checking the convergence
convergence = false;
agentReached2Goal = false;
agentBumped2wall = false;
%% Episode Loops
ei = 1;
while (ei<maxNumEpisodes && ~convergence), % ei<maxNumEpisodes && % ei is counter for episodes
     % initialize the starting state - Continuous state
     s = [-0.5,0.0];
     x = s(1); y = s(2);
     
     if strcmp(functionApproximator,'Qtable'),
             [~,idx] = min(dist(x,xInputInterval));
             [~,idy] = min(dist(y,yInputInterval));
             sti = sub2ind([length(xInputInterval),length(yInputInterval)],idx,idy);
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
             [Q,h,id] = kwta_NN_forward(st,Wih,biasih,Who,biasho);
         case 'regularBPNN',
             [Q,h] = regularBPNN_forward(st, Wih,biasih, Who,biasho);
         case 'linearNN'
              Q  = SimpleNN_forward(st,Wio,biasio);
         case 'Qtable'
              Q = Qtable(sti,:);
     end
     act = e_greedy_policy(Q,nActions,epsilon);
     % deltaForStepsOfEpisode = [];

    %% Episode While Loop
    while( ~agentReached2Goal && ts<maxIteratonEpisode ),
        % update state to state+1
        sp1 = MountainCar_updateState(s,act,xVector,yVector);
        xp1 = sp1(1); yp1 = sp1(2);
        
        switch functionApproximator,
            case 'Qtable',
                [~,idxp1] = min(dist(xp1,xInputInterval));
                [~,idyp1] = min(dist(yp1,yInputInterval));
                stp1i = sub2ind([length(xInputInterval),length(yInputInterval)],idxp1,idyp1);
        end
        
        xtp1 = sigmax * sqrt(2*pi) * normpdf(xInputInterval,xp1,sigmax);
        ytp1 = sigmay * sqrt(2*pi) * normpdf(yInputInterval,yp1,sigmay);
        stp1=[xtp1,ytp1];
        
        if ( sp1(1)>=s_end ),
            agentReached2Goal = true;
        else
            agentReached2Goal = false;            
        end
        
        % reward/punishment from Environment
        rew = MontainCarENV_REWARD(agentReached2Goal,nTilex,nTiley);
        
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
    
        if( ~agentReached2Goal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
        else
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            fprintf('Reaching to Goal at episode =%d at step = %d and mean(delta) = %f \n',ei,ts,mean(deltaForStepsOfEpisode));
            delta = rew - Q(act);
        end
        deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
           
            % Update Neural Net
           switch functionApproximator,
               case 'kwtaNN',
                   [Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho); 
               case 'regularBPNN',
                   [Wih,biasih,Who,biasho] = Update_regularBPNN(st,act,h,alpha,delta,Wih,biasih,Who,biasho,withBias);
               case 'linearNN'
                   [Wio,biasio] = UpdateSimpleNN(st,act,alpha,delta,Wio,biasio);
               case 'Qtable'
                    Qtable(sti,act) = Qtable(sti,act) + alphaTable * delta; 
           end
           if agentReached2Goal, break; end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1;
        Q = Qp1;    
        if strcmp(functionApproximator,'Qtable'), sti = stp1i; end
        if strcmp(functionApproximator,'regularBPNN'), h = hp1; end
        if strcmp(functionApproximator,'kwtaNN'), id = idp1; h = hp1; end
        ts = ts + 1;
    end % while loop
    meanDeltaForEpisode(ei) = mean(deltaForStepsOfEpisode);
    varianceDeltaForEpisode(ei) =var(deltaForStepsOfEpisode);
    stdDeltaForEpisode(ei) = std(deltaForStepsOfEpisode);
    

    if ( ei>500 && abs(meanDeltaForEpisode(ei))< 0.2 && agentReached2Goal ),
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
    
    
    plot(meanDeltaForEpisode)      
    title(['Episode: ',int2str(ei),' epsilon: ',num2str(epsilon)])    
    drawnow
    
    ei = ei + 1;

end  % end episode loop

%% Save Variables
data.meanDeltaForEpisode = meanDeltaForEpisode;
data.varianceDeltaForEpisode = varianceDeltaForEpisode;
data.stdDeltaForEpisode = stdDeltaForEpisode;
weights.Wih = Wih;
weights.biasih = biasih;
weights.Who = Who;
weights.biasho = biasho;
weights.biasio = biasio;
weights.Wio = Wio;
weights.Qtable = Qtable;