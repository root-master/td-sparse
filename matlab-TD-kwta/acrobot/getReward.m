function [rew] = getReward(agentReachedGoal)

if agentReachedGoal,
    rew = 0.0;
else
    rew = -1.0;
end