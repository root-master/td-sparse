function [cost] = plotQ(weights,functionApproximator,nMeshp,nMeshv,grafica)

Wih = weights.Wih;
biasih = weights.biasih;
Who = weights.Who;
biasho = weights.biasho;
Wio = weights.Wio;
biasio = weights.biasio;
Qtable = weights.Qtable;

nTilep = 1;
nTilev = 1;

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

% pVector = positionBound(1) : dp : positionBound(2);
pVector = positionBound(1) : (dp/nTilep) : 0.6;
vVector = velocityBound(1) : (dv/nTilev) : velocityBound(2);

nStates = length(Qtable);
statesList = zeros(nStates,2);
index = 1;
for i = 1:length(inputPVector)
    for j = 1:length(inputVVector)
        statesList(index,:) = [inputPVector(i),inputVVector(j)];
        index = index + 1;
    end
end

sigmap = dp/nTilep;
sigmav = dv/nTilev;

Qestimate = zeros(length(pVector),length(vVector));
for i = 1:length(pVector)
    for j = 1: length(vVector)
        p = pVector(i); v = vVector(j); s = [p,v];
        [~,sti] = min(dist(statesList,s'));

        % Gaussian Distribution on continuous state
        
        
        pt = sigmap * sqrt(2*pi) * normpdf(inputPVector,p,sigmap);
        vt = sigmav * sqrt(2*pi) * normpdf(inputVVector,v,sigmav);
        % Using st as distributed input for function approximator
        st = [pt,vt];    
        switch functionApproximator,
            case 'kwtaNN',
                [Q,~,~] = kwta_NN_forward(st,Wih,biasih,Who,biasho);
                [Qestimate(i,j),~] = max(Q);
                
            case 'regularBPNN',
                [Q,~] = regularBPNN_forward(st, Wih,biasih, Who,biasho);
                [Qestimate(i,j),~] = max(Q);
            case 'linearNN',
                Q = SimpleNN_forward(st,Wio,biasio);
                [Qestimate(i,j),~] = max(Q);
            case 'Qtable',
                Q = Qtable(sti,:);
                [Qestimate(i,j),~] = max(Q);
        end
        if p>=goalPosition,
            Qestimate(i,j) = 0;
        end
    end
end
cost = -Qestimate;
if grafica,
    surf(pVector,vVector,cost');
end
