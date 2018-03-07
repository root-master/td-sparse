function [Qestimate] = plotQ_PolicyMap(s_end,nMeshx,xVector,xInputInterval,nMeshy,yVector,yInputInterval,weights,functionApproximator,grafica)
%% Weights of NN
Wih = weights.Wih;
Who = weights.Who;
biasih = weights.biasih;
biasho = weights.biasho;
Wio = weights.Wio;
biasio = weights.biasio;
switch functionApproximator,
    case 'Qtable',
        Qtable = weights.Qtable;
end
% parameter of Gaussian Distribution
sigmax = 1.0 / nMeshx; 
sigmay = 1.0 / nMeshy;

policy = ones(length(xInputInterval),length(yInputInterval));
Qestimate = zeros(length(xInputInterval),length(yInputInterval));

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

if grafica,
%% plotting
	figure;
	% plot enviroment
	plotEnvironment(nMeshx,nMeshy);
	% plot goal
	plot(s_end(2),s_end(1),'p','MarkerSize',20,'MarkerFaceColor','g'); 
end
%% optimal actions and costs for all states
directionNotation = ['^' 'v' '>' '<']; % Up, Down, Right, Left
directionColor = ['k' 'k' 'k' 'k'];

for i=1:length(xInputInterval),
    for j=1:length(yInputInterval),
        x = xInputInterval(i); y = yInputInterval(j); s=[x,y];

        % Gaussian Distribution on continuous state
        xt = sigmax * sqrt(2*pi) * normpdf(xInputInterval,x,sigmax);
        yt = sigmay * sqrt(2*pi) * normpdf(yInputInterval,y,sigmay);
        % Using st as distributed input for function approximator
        st = [xt,yt];     
        switch functionApproximator,
            case 'kwtaNN',
                [Q,~,~] = kwta_NN_forward(st,shunt,Wih,biasih,Who,biasho);
                [Qestimate(i,j),policy(i,j)] = max(Q);
            case 'allHiddenUnitsForwardButNoErrorForLosers'
                [Q,~,~] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
                [Qestimate(i,j),policy(i,j)] = max(Q);
            case 'LosersForwardZeroButErrorForAll'
                [Q,~,~] = kwta_NN_forward(st,shunt,Wih,biasih,Who,biasho);
                [Qestimate(i,j),policy(i,j)] = max(Q);
            case 'allHiddenUnitsForwardAndAllGetErrors'
                [Q,~,~] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
                [Qestimate(i,j),policy(i,j)] = max(Q);
            case  'allHiddenUnitsForwardAndAllGetErrorsShunt1'
                [Q,~,~] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
                [Qestimate(i,j),policy(i,j)] = max(Q);
            case 'regularBPNN',
                [Q,~] = regularBPNN_forward(st, Wih,biasih, Who,biasho);
                [Qestimate(i,j),policy(i,j)] = max(Q);
            case 'linearNN',
                Q = SimpleNN_forward(st,Wio,biasio);
                [Qestimate(i,j),policy(i,j)] = max(Q);
            case 'Qtable',
        sti = sub2ind([length(xInputInterval),length(yInputInterval)],i,j);
        [Qestimate(i,j),policy(i,j)] = max(Qtable(sti,:));
        
        end
        % plot policy map
		if grafica && ~(  i==length(xInputInterval) && j == length(xInputInterval)),
			plot(x,y,directionNotation(policy(i,j)),'MarkerSize',10,'MarkerFaceColor',directionColor(policy(i,j)));
		end
    end
end


xlabel('$x$'); ylabel('$y$');
title('Policy map')
box on

if strcmp(functionApproximator,'Qtable'), 
cost = -Qestimate;
    if grafica,
	% plot cost
    figure,
	surf(xVector,yVector,cost');
    colormap hot;
    colormap(flipud(colormap))
    grid on
	title('Estimate of Value Function')
	xlabel('$x$')
	ylabel('$y$')
	zlabel('$-Q_{max}$')
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nTilex = 3; nTiley = 3;
% parameter of Gaussian Distribution
sigmax = 1.0 / nMeshx; 
sigmay = 1.0 / nMeshy;
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;
% smoother state space with tiling
xgrid = 1 / (nMeshx * nTilex);
ygrid = 1 / (nMeshy * nTiley);
xVector = 0:xgrid:1;
yVector = 0:ygrid:1;
QestimatePlot = zeros(length(xVector),length(yVector));

for i=1:length(xVector),
    for j=1:length(yVector),
        x = xVector(i); y = yVector(j);
        % Gaussian Distribution on continuous state
        xt = sigmax * sqrt(2*pi) * normpdf(xInputInterval,x,sigmax);
        yt = sigmay * sqrt(2*pi) * normpdf(yInputInterval,y,sigmay);
        % Using st as distributed input for function approximator
        st = [xt,yt];     
        switch functionApproximator,
            case 'kwtaNN'
                [Q,~,~] = kwta_NN_forward(st,shunt,Wih,biasih,Who,biasho);
                [QestimatePlot(i,j),~] = max(Q);
            case 'allHiddenUnitsForwardButNoErrorForLosers'
                [Q,~,~] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
                [QestimatePlot(i,j),~] = max(Q);
            case 'LosersForwardZeroButErrorForAll'
                [Q,~,~] = kwta_NN_forward(st,shunt,Wih,biasih,Who,biasho);
                [QestimatePlot(i,j),~] = max(Q);
            case 'allHiddenUnitsForwardAndAllGetErrors'
                [Q,~,~] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
                [QestimatePlot(i,j),~] = max(Q);
            case  'allHiddenUnitsForwardAndAllGetErrorsShunt1'
                [Q,~,~] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
                [QestimatePlot(i,j),~] = max(Q);
            case 'regularBPNN'
                [Q,~] = regularBPNN_forward(st, Wih,biasih, Who,biasho);
                [QestimatePlot(i,j),~] = max(Q);
            case 'linearNN'
                 Q = SimpleNN_forward(st,Wio,biasio);
                 [QestimatePlot(i,j),~] = max(Q);
        end
    end
end

if ~strcmp(functionApproximator,'Qtable'),
cost = -QestimatePlot;
if grafica,
	% plot cost
    figure,
	surf(xVector,yVector,cost');
    colormap hot;
    colormap(flipud(colormap))
    grid on
	title('Estimate of Value Function')
	xlabel('$x$')
	ylabel('$y$')
	zlabel('$-Q_{max}$')
end
end