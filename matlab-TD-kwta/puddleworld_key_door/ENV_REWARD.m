function rew = ENV_REWARD(s)
% convert to row/column notation: 
[agentinPuddle,dist2Edge] = CreatePuddle(s);
rew = 0;
if agentinPuddle,
    rew = min(-400.0 * dist2Edge,-1.0);
end