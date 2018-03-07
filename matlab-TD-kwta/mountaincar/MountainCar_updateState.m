function [sp1] = MountainCar_updateState(s,action,pVector,vVector)

sp1 = [0.0,0.0];
%% [forward=+1, neutral=0, backward=-1]

oldPosition = s(1); % continues position state
oldVelocity = s(2); % continues velocity state

newVelocity = oldVelocity + 0.001 * action + (-0.0025 * cos( 3.0 * oldPosition) );
% newVelocity = newVelocity * 0.999;

newVelocity = bound( newVelocity , vVector );

newPosition = oldPosition + newVelocity; 

if(newPosition <= pVector(1) )
    newPosition = pVector(1);
    newVelocity = 0.0;
end

sp1(1) = newPosition;
sp1(2) = newVelocity;

% % bound position inside the grid:
% sp1(1) = bound( sp1(1) , pVector );
% sp1(2) = bound( sp1(2) , vVector );
