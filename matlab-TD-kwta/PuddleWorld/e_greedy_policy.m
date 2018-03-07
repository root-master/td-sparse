function action = e_greedy_policy(Q,nActions,epsilon)
% pick action using an epsilon greedy policy derived from Q: 
 
  % act \in [1,2,3,4]=[up,down,right,left]
  if( rand<epsilon || norm(Q)==0 ),        
      % explore ... with a random action 
       action=randi(nActions); 
  else 
      [~,action] = max(Q); 
  end