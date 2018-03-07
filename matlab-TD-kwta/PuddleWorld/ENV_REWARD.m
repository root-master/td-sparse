function rew = ENV_REWARD(s,agentReached2Goal,agentBumped2wall,numTilex,numTiley)
% convert to row/column notation: 
[agentinPuddle,dist2Edge] = CreatePuddle(s);
numTiles = max(numTilex,numTiley);
if agentReached2Goal,
    rew = 0;
elseif agentinPuddle,
    rew = min(-400 * dist2Edge,-1);
elseif agentBumped2wall,
    rew = -2;
else
    rew = -1;
end
rew = rew / numTiles;