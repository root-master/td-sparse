function [successful_key_door_episodes, successful_key_episodes, scores_vec, total_episodes] = test_score_success_func(ep,Wih, biasih, Who, biasho)

nMeshx = 10; nMeshy = 10;

successful_key_door_episodes = [];
successful_key_episodes = [];
scores_vec = [];
% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;

xgrid = 1 / (nMeshx);
ygrid = 1 / (nMeshy);
% parameter of Gaussian Distribution
sigmax = 1.0 / nMeshx; 
sigmay = 1.0 / nMeshy;

ep_id = 1;
max_iter = 1000;
total_episodes = 0;

keyinPuddle = true;
while keyinPuddle
    key = initializeState(xInputInterval,yInputInterval);
    [keyinPuddle,~] = CreatePuddle(key);
end
%fprintf('key = %g %g \n',key);

doorinPuddle = true;
while doorinPuddle
    door = initializeState(xInputInterval,yInputInterval);
    [doorinPuddle,~] = CreatePuddle(door);
end

%fprintf('door = %g %g \n',door);


for x=xInputInterval,
    for y=yInputInterval,
        t = 1;
        scores = 0;
        agenthaskey = false;
        g = key;
        s=[x,y];
        [agentinPuddle,~] = CreatePuddle(s);
        if agentinPuddle
            continue
        end
        %fprintf('s0 = %g %g and goal = %g %g \n',s,g);
        while(t<=max_iter)
            %fprintf('s = %g %g \n',s);
            if success(s,key) && ~agenthaskey
                scores = scores + 10;
                g = door;
                %fprintf('goal changed to %g %g \n',g);
                successful_key_episodes = [successful_key_episodes, ep_id];
                agenthaskey = true;
            end
            
            if agenthaskey
               if success(s,door)
                   agentReached2Door = true;
                   scores = scores + 100;
                   successful_key_door_episodes = [successful_key_door_episodes, ep_id];
                   scores_vec = [scores_vec, scores];
                   %fprintf('goal acheived \n');
                   break
               end
            end
            
            
             sx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,s(1),sigmax);
             sy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,s(2),sigmay);
             gx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,g(1),sigmax);
             gy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,g(2),sigmay);
             % Using st as distributed input for function approximator
             st = [sx,sy,gx,gy];                
             Q = kwta_NN_forward_new(st, Wih, biasih, Who, biasho);
             [~,a] = max(Q);
             sp1 = UPDATE_STATE(s,a,xgrid,xInputInterval,ygrid,yInputInterval);
             rew = ENV_REWARD(sp1);
             scores = scores + rew;
             
             s = sp1;
             t = t+1;
            if t == max_iter
                scores_vec = [scores_vec, scores];
            end
        end
                
        ep_id = ep_id + 1;
        total_episodes = total_episodes + 1;
    end
end
