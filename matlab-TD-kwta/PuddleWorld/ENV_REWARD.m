function rew = ENV_REWARD(s,agentReached2Goal,agentBumped2wall)
% convert to row/column notation: 
[agentinPuddle,dist2Edge] = CreatePuddle(s);
if agentReached2Goal,
    rew = 0;
elseif agentinPuddle,
    rew = min(-400 * dist2Edge,-2);
elseif agentBumped2wall,
    rew = -2;
else
    rew = -1;
end
