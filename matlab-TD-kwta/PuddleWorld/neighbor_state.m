function g = neighbor_state(s0,xVector,yVector,radius)

g = s0;

[s0inpuddle,dist_2_edge] = CreatePuddle(s0);
% if s0inpuddle
%     goalinPuddle = true;
%     while (goalinPuddle),
%         g = initializeState(xVector,yVector);
%         [goalinPuddle,~] = CreatePuddle(g);
%     end
%     return
% end


if s0inpuddle
    eff_radius = dist_2_edge + radius + 0.5;
else
    eff_radius = radius;
end

x0 = s0(1);
y0 = s0(2);
x_close_vec = xVector( abs(xVector - x0) < eff_radius );
y_close_vec = yVector( abs(yVector - y0) < eff_radius );

goalinPuddle = true;

while (goalinPuddle),
    gi = randi(length(x_close_vec));
    gj = randi(length(y_close_vec));
    g =  [x_close_vec(gi),y_close_vec(gj)];
    [goalinPuddle,~] = CreatePuddle(g);
end


