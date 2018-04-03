function x = bound(x , xInterval)
leftbound = xInterval(1);
rightbound = xInterval(end);
if x < leftbound 
    x = leftbound; 
end

if x > rightbound,
    x = rightbound;
end