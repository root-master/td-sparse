function st = initializeState(xVector,yVector)
% random state (x,y) inside gridworld
x = xVector ( randi( length(xVector) ) );
y = yVector ( randi( length(yVector) ) );
st = [x,y];