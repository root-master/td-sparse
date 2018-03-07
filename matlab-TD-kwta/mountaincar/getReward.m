function [rew,agentReachedGoal] = getReward(statep1,goalPosition)
% MountainCarGetReward returns the reward at the current state
% stp1: a vector of index of [position, velocity] of the car
% reward: the returned reward.

%goalArea = goalPosition - 0.05;
% i changed to 
goalArea = goalPosition;

if( statep1(1) >= goalArea  ) 
	rew = 0;
    agentReachedGoal = true;
else
    rew = -1;
    agentReachedGoal = false;
end


    
   


    
