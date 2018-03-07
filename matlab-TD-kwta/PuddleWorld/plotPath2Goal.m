function [discReward,agentReached2Goal] = plotPath2Goal(s_0,s_end,xgrid,nMeshx,xVector,xInputInterval,sigmax,nMeshy,ygrid,yVector,yInputInterval,sigmay,weights,functionApproximator,grafica)
%% Weights of NN
Wih = weights.Wih;
Who = weights.Who;
biasih = weights.biasih;
biasho = weights.biasho;
nTilex = 1; nTiley = 1;
switch functionApproximator,
    case 'linearNN',
        Wio = weights.Wio;
        biasio = weights.biasio;
    case 'Qtable',
        Qtable = weights.Qtable;
end
discReward = 0;
gamma = 0.99;
%% plotting
if grafica,
figure;
box on

% plot enviroment
plotEnvironment(nMeshx,nMeshy);
% plot starting state
plot(s_0(1),s_0(2),'o','MarkerSize',10,'MarkerFaceColor','b');
% plot goal
plot(s_end(2),s_end(1),'p','MarkerSize',20,'MarkerFaceColor','g'); 
xlabel('$x$');
ylabel('$y$');
title('Puddle world task');
grid on
end
%% optimal path from s_0 to s_end 
directionNotation = ['^' 'v' '>' '<']; % Up, Down, Right, Left
s = s_0;
x = s(1); y = s(2);     
pathx = [x]; pathy = [y];     
% Gaussian Distribution on continuous state
xt = sigmax * sqrt(2*pi) * normpdf(xInputInterval,x,sigmax);
yt = sigmay * sqrt(2*pi) * normpdf(yInputInterval,y,sigmay);
% Using st as distributed input for function approximator
st = [xt,yt];     
% initializing time
ts = 1;
agentReached2Goal = false;

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

while( ~agentReached2Goal && ts < 2*(length(xVector)+length(yVector)) ),
    switch functionApproximator,
        case 'kwtaNN'
             [Q,~,~] = kwta_NN_forward(st,shunt,Wih,biasih,Who,biasho);
        case 'allHiddenUnitsForwardButNoErrorForLosers'
             [Q,~,~] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
        case 'LosersForwardZeroButErrorForAll'
             [Q,~,~] = kwta_NN_forward(st,shunt,Wih,biasih,Who,biasho);
        case 'allHiddenUnitsForwardAndAllGetErrors'
             [Q,~,~] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
        case  'allHiddenUnitsForwardAndAllGetErrorsShunt1'
             [Q,~,~] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
        case 'regularBPNN'
            [Q,~] = regularBPNN_forward(st, Wih,biasih, Who,biasho);
        case 'linearNN'
            Q = SimpleNN_forward(st,Wio,biasio);
        case 'Qtable',
            [~,idx] = min(dist(x,xInputInterval));
            [~,idy] = min(dist(y,yInputInterval));
            sti = sub2ind([length(xInputInterval),length(yInputInterval)],idx,idy);
            Q = Qtable(sti,:);
    end
    [~,act] = max(Q);
    % update state to state+1
    sp1 = UPDATE_STATE(s,act,xgrid,xVector,ygrid,yVector);
    x = sp1(1); y = sp1(2);
    if ( sp1(1)==s_end(1) && s(2)==s_end(2) ),
        agentReached2Goal = true;
        agentBumped2wall = false;
    elseif ( sp1(1)==s(1) && sp1(2)==s(2) ),
        agentBumped2wall = true;
        agentReached2Goal = false;
    else
        agentBumped2wall = false;
        agentReached2Goal = false;            
    end
    rew = ENV_REWARD(sp1,agentReached2Goal,agentBumped2wall,nTilex,nTiley);

    discReward = discReward + gamma^(ts-1) * rew; 
    if grafica,
        plot(x,y,directionNotation(act),'MarkerSize',10,'MarkerFaceColor','k');
    end
    pathx = [pathx,x]; pathy = [pathy,y];
    xt = sigmax * sqrt(2*pi) * normpdf(xInputInterval,x,sigmax);
    yt = sigmay * sqrt(2*pi) * normpdf(yInputInterval,y,sigmay);
    st = [xt,yt]; s = sp1; 
    ts = ts + 1; 
end % while loop
if grafica,
   
    plot(pathx,pathy,'-k','LineWidth',3);
end