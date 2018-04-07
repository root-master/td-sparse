function [successful_key_door_episodes, successful_key_episodes, scores_vec, total_episodes] = test_score_success_func_2_hidden(Wih, biasih,Wh1h2,biash1h2, Who, biasho)

nMeshx = 20; nMeshy = 20;

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
max_iter = 300;
total_episodes = 0;

keyinPuddle = true;
while keyinPuddle
    key = initializeState(xInputInterval,yInputInterval);
    [keyinPuddle,~] = CreatePuddle(key);
end
fprintf('key = %g %g \n',key);

doorinPuddle = true;
while doorinPuddle
    door = initializeState(xInputInterval,yInputInterval);
    [doorinPuddle,~] = CreatePuddle(door);
end
fprintf('door = %g %g \n',door);


for x=xInputInterval,
    for y=yInputInterval,
        agentReached2Key = false;
        agentReached2Door = false;
        t = 1;
        scores = 0;
        first_time_visit_key = false;
        
        s=[x,y];
        [agentinPuddle,~] = CreatePuddle(s);
        if agentinPuddle
            %continue
        end
        
        g = key;
        fprintf('s0 = %g %g and goal = %g %g \n',s,g);
        while(t<=max_iter)
            fprintf('s = %g %g \n',s);
            if success(s,key) && ~first_time_visit_key
                agentReached2Key = true;
                scores = scores + 10;
                g = door;
                fprintf('goal changed to door at %g %g \n',g);
                successful_key_episodes = [successful_key_episodes, ep_id];
                first_time_visit_key = true;
            end
            
            if agentReached2Key
               if success(s,door)
                   agentReached2Door = true;
                   scores = scores + 100;
                   successful_key_door_episodes = [successful_key_door_episodes, ep_id];
                   scores_vec = [scores_vec, scores];
                   fprintf('goal acheived \n');
                   break
               end
            end
            
            
             sx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,s(1),sigmax);
             sy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,s(2),sigmay);
             gx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,g(1),sigmax);
             gy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,g(2),sigmay);
             % Using st as distributed input for function approximator
             st = [sx,sy,gx,gy];                
             [Q,~,~,~,~]  = kwta_NN_forward_2_layer(s, Wih,biasih, Wh1h2,biash1h2, Who,biasho);
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
