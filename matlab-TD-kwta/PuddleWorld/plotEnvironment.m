function [] = plotEnvironment(nMeshx,nMeshy)
radius = 0.1; % radius of puddle semi-circles at end
% number of tiles is set to 10 in order to have an smooths plot of puddle
nTilex = 10;
nTiley = 10;

xgrid = 1 / (nMeshx * nTilex);
ygrid = 1 / (nMeshy * nTiley);

xVector = 0:xgrid:1;
yVector = 0:ygrid:1;

set(gcf,'name','Puddle World Task')  
set(gco,'Units','uniform')
axis([0 1.0 0 1.0]);
hold on
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
hold on
for i=1:length(xVector),
    for j=1:length(yVector),
        x = xVector(i); y = yVector(j); s = [x,y];
        [inPuddle,dist2Edge] = CreatePuddle(s);
        if inPuddle,
            if (dist2Edge >= 0.7 * radius),
                puddleMarkerColor = 'r';
                puddlemarrkerSize = 8;
            elseif ( dist2Edge<0.7*radius && dist2Edge >0.4*radius ),
                puddleMarkerColor = 'r';
                puddlemarrkerSize = 6;
            else
                puddleMarkerColor = 'r';
                puddlemarrkerSize = 4;
            end
            plot(x,y,'.','MarkerSize',puddlemarrkerSize,'LineWidth',6,'MarkerEdgeColor',puddleMarkerColor);   
        end
    end
end