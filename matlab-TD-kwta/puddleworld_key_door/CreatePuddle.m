function [agentinPuddle,closestDist] = CreatePuddle(s)
% this function gets the coordinate s=(x,y) state and declares if s is inside
% the puddle and if so calculate the distance of s from closest edge of puddle
%s=[0.46,0.81]
x = s(1); y = s(2);
closestDist = 0;
% Horizontal wing of puddle consists of 
% 1) rectangle area xch1<= x <=xc2 && ych1-radius <= y <=ych2+radius
% where (xchi,ychi) is the center points (h ==> horizantal)
% 1 : left; 2 : right
xch1 = 0.3; ych1=0.7;
xch2 = 0.65; ych2 = ych1;
radius = 0.1;
inHorRec = (x>=xch1) && (y>= ych1-radius) && (x<=xch2) && (y<=ych2+radius);    
% 2) two half-circle at end edges of rectangle
inHorCir1 = ( ( (x-xch1)^2 + (y-ych1)^2 <= radius^2 ) && x<xch1 );
inHorCir2 = ( ((x-xch2)^2 + (y-ych2)^2) <= radius^2 && x>xch2 );
inHor = inHorRec || inHorCir1 || inHorCir2;

% Vertical wing of puddle consists of 
% 1) rectangle area xcv1-radius<= x <=xcv2+radius && ycv1 <= y <= ycv2
% where (xcvi,ycvi) is the center points (v ==> vertical)
% 1 : left; 2 : right
xcv1 = 0.45; ycv1=0.4;
xcv2 = xcv1; ycv2 = 0.8;

inVerRec = (x >= xcv1-radius) && (y >= ycv1) && (x <= xcv2+radius) && (y <= ycv2);    
% 2) two half-circle at end edges of rectangle
inVerCir1 = ( ( (x-xcv1)^2 + (y-ycv1)^2 <= radius^2 ) && y<ycv1 );
inVerCir2 = ( ( (x-xcv2)^2 + (y-ycv2)^2 <= radius^2 ) && y>ycv2 );
inVer = inVerRec || inVerCir1 || inVerCir2;

agentinPuddle = inHor || inVer;

% for semi-circle parts
numGrids = 20;
theta = 0:pi/numGrids:pi;
if inHorCir1,
    xp = xch1 + radius * cos(theta+pi/2);
    yp = ych1 + radius * sin(theta+pi/2);
    points = [xp;yp]';
    closestDistHorCircle = min(dist(points,s'));
end

if inHorCir2,
    xp = xch2 + radius * cos(theta-pi/2);
    yp = ych2 + radius * sin(theta-pi/2);
    points = [xp;yp]';
    closestDistHorCircle = min(dist(points,s'));
end

if inVerCir1,
    xp = xcv1 + radius * cos(theta+pi);
    yp = ycv1 + radius * sin(theta+pi);
    points = [xp;yp]';
    closestDistVerCircle = min(dist(points,s'));
end

if inVerCir2,
    xp = xcv1 + radius * cos(theta);
    yp = ycv2 + radius * sin(theta);
    points = [xp;yp]';
    closestDistVerCircle = min(dist(points,s'));
end

% to find the distance from edges of puddle we consider possible cases
% for a point inside puddle

if inHor && ~inVer,
    if inHorRec,
        points = [x,ych1+radius;x,ych1-radius];
        closestDist = min(dist(points,s'));
    else
        closestDist = closestDistHorCircle;
    end
elseif inHorRec && inVerRec,
    points = [xcv1-radius,ych1-radius;xcv1-radius,ych1+radius;xcv1+radius,ych1-radius;xcv1+radius,ych2+radius];
    closestDist = min(dist(points,s'));    
elseif inHorRec && (inVerCir1 || inVerCir2),
    points = [x,ych1+radius;x,ych1-radius];
    closestDist = min(dist(points,s'));
    closestDist = max(closestDist,closestDistVerCircle);
end     

if inVer && ~inHor,
    if inVerRec
        points = [xcv1-radius,y;xcv1+radius,y];
        closestDist = min(dist(points,s'));
    else
        closestDist = closestDistVerCircle;
    end
elseif inVerRec && (inHorCir1 || inHorCir2),
    points = [xcv1-radius,y;xcv1+radius,y];
    closestDist = min(dist(points,s'));
    closestDist = max(closestDist,closestDistHorCircle);
elseif (inVerCir1 || inVerCir2) && (inHorCir1 || inHorCir2),
    closestDist = max(closestDistHorCircle,closestDistVerCircle);    
end