function [action] = Softmax_Policy(Q,nActions,lambda)
% pick action using an epsilon greedy policy derived from Q: 
 
  % act \in [1,2,3,4]=[up,down,right,left]
  
  % Probibility
  P = exp(lambda * Q) / sum(exp(lambda * Q));
  
  if( lambda  == 0)        
      % explore ... with a random action 
      action=randi(nActions); 
  else 
      [~,action] = max(P); 
  end