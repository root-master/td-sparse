function sp1 = UPDATE_STATE(s,act,xgrid,xVector,ygrid,yVector)

% state st and action to st+1
%   

% convert to row/column notation: 
x = s(1); y = s(2); 

% incorporate any actions and fix our position if we end up outside the grid:
% 
switch act
 case 1, % action = UP 
     sp1 = [x,y+ygrid];
 case 2, % action = DOWN
     sp1 = [x,y-ygrid];
 case 3, % action = RIGHT
     sp1 = [x+xgrid,y];
 case 4  % action = LEFT 
     sp1 = [x-xgrid,y];
end

% bound position inside the grid:
sp1(1) = bound( sp1(1) , xVector );
sp1(2) = bound( sp1(2) , yVector );
