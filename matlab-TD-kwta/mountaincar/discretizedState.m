function sti = discretizedState( state, statesList )
% the closest digital state to the continues state

[~ , sti] = min(dist(statesList,state'));

