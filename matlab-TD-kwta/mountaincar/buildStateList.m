function [statesList,positionVector,velocityVector] = buildStateList(positionBound,velocityBound,nMeshPosition,nMeshVelocity,goalPosition)

% Compute the mesh (discrete grid) widths: 
dp = diff( positionBound ) / nMeshPosition;
dv = diff( velocityBound ) / nMeshVelocity; 

% Discritized position and velocity 
positionVector = positionBound(1) : dp : goalPosition;
velocityVector = velocityBound(1) : dv : velocityBound(2);
nStates = length(positionVector) * length(velocityVector);
statesList = zeros(nStates,2);

index = 1;
for i = 1:length(positionVector)
    for j = 1:length(velocityVector)
        %statesList( sub2ind( [length(positionVector),length(velocityVector)] ,i,j) , :) = [positionVector(i),velocityVector(j)];
        statesList(index,:) = [positionVector(i),velocityVector(j)];
        index = index + 1;
    end
end