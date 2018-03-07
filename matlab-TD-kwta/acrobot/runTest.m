function [ts,deltaForStepsOfEpisode,capture,error,F] = runTest(nMesh,functionApproximator,weights,Qtable,statesList,grafica)

if grafica,
    figure,
    set(gcf,'BackingStore','off')  % for realtime inverse kinematics
    set(gco,'Units','data')
    set(gcf,'name','SARSA Reinforcement Learning Acrobot');
    set(gcf,'Color','w') 
end
error = [];
Wih = weights.Wih;
Who = weights.Who;
biasih = weights.biasih;
biasho = weights.biasho;
Wio = weights.Wio;
biasio = weights.biasio;
Qtable = weights.Qtable;

theta1_bound = [-pi , pi]; 
theta2_bound = [-pi , pi];
omega1_bound = [-4*pi , 4*pi]; 
omega2_bound = [-9*pi,9*pi];

% number of mesh for descritzing the state space
nMesh_theta1 = nMesh(1);
nMesh_omega1 = nMesh(2);
nMesh_theta2 = nMesh(3);
nMesh_omega2 = nMesh(4);

dtheta1 = diff( theta1_bound ) / nMesh_theta1;
dtheta2 = diff( theta2_bound ) / nMesh_theta2;
domega1 = diff( omega1_bound ) / nMesh_omega1;
domega2 = diff( omega2_bound ) / nMesh_omega2;

% input vector corressoponding to state variables 
theta1_vec = theta1_bound(1) : dtheta1 : theta1_bound(2);
theta2_vec = theta2_bound(1) : dtheta2 : theta2_bound(2);
omega1_vec = omega1_bound(1) : domega1 : omega1_bound(2);
omega2_vec = omega2_bound(1) : domega2 : omega2_bound(2);

actionsList = [-1, 0, 1]; % apply torque to [backward, neutral?, forward]
nActions = length(actionsList); % number of actions 

sigma_theta1 = dtheta1;
sigma_theta2 = dtheta2;
sigma_omega1 = domega1;
sigma_omega2 = domega2;
gamma = 0.99;
epsilon = 0;

maxIteratonEpisode = 300;
capture = [0,0,0,0]';
% F = struct('cdata',[],'colormap',[]);

% initialize the starting state - Continuous state
     s = [0.0,0.0,0.0,0.0];
     % s = initializeState(theta1_vec,omega1_vec,theta2_vec,omega2_vec);
     theta1 = s(1); omega1 = s(2); 
     theta2 = s(3); omega2 = s(4); 
     deltaForStepsOfEpisode = [];
     sti = dsearchn(statesList,s); 
     
     % Gaussian Distribution on continuous state
     theta1_input = sigma_theta1 * sqrt(2*pi) * normpdf(theta1_vec,theta1,sigma_theta1);
     omega1_input = sigma_omega1 * sqrt(2*pi) * normpdf(omega1_vec,omega1,sigma_omega1);
     theta2_input = sigma_theta2 * sqrt(2*pi) * normpdf(theta2_vec,theta2,sigma_theta2);
     omega2_input = sigma_omega2 * sqrt(2*pi) * normpdf(omega2_vec,omega2,sigma_omega2);

     % Using st as distributed input for function approximator
     st = [theta1_input,omega1_input,theta2_input,omega2_input];     
     
     % initializing time
     ts = 1;
     switch functionApproximator,
         case 'kwtaNN'
             [Q,~,~] = kwta_NN_forward(st,Wih,biasih,Who,biasho);
         case 'regularBPNN',
             [Q,~] = regularBPNN_forward(st,Wih,biasih, Who,biasho);
         case 'linearNN'
              Q  = SimpleNN_forward(st,Wio,biasio);
         case 'Qtable'
              Q = Qtable(sti,:);
     end
     
     Qt = Qtable(sti,:);
     
     
     
     act = e_greedy_policy(Q,nActions,epsilon);
     agentReachedGoal = false;

 %% Episode While Loop
    while(  ts<maxIteratonEpisode ),
        % convert the index of the action into an action value
        action = actionsList(act); % action = [-1,0,+1]
        % update state to state+1
        [sp1,agentReachedGoal] = updateState(s,action,omega1_bound,omega2_bound);
        % reward/punishment from Environment
        [rew] = getReward(agentReachedGoal);
        
        theta1_p1 = sp1(1); omega1_p1 = sp1(2);
        theta2_p1 = sp1(3); omega2_p1 = sp1(4); 
        
        stp1i = dsearchn(statesList,sp1);

        theta1_p1_input = sigma_theta1 * sqrt(2*pi) * normpdf(theta1_vec,theta1_p1,sigma_theta1);
        omega1_p1_input = sigma_omega1 * sqrt(2*pi) * normpdf(omega1_vec,omega1_p1,sigma_omega1);
        theta2_p1_input = sigma_theta2 * sqrt(2*pi) * normpdf(theta2_vec,theta2_p1,sigma_theta2);
        omega2_p1_input = sigma_omega2 * sqrt(2*pi) * normpdf(omega2_vec,omega2_p1,sigma_omega2);

        stp1=[theta1_p1_input,omega1_p1_input,theta2_p1_input,omega2_p1_input];
        
        switch functionApproximator,
            case 'kwtaNN',
                [Qp1,hp1,idp1] = kwta_NN_forward(stp1,Wih,biasih,Who,biasho);
            case 'regularBPNN',
                [Qp1,hp1] = regularBPNN_forward(stp1, Wih,biasih, Who,biasho);
            case 'linearNN'
                 Qp1  = SimpleNN_forward(stp1,Wio,biasio);
            case 'Qtable'
                 Qp1 = Qtable(stp1i,:);
        end
        
        Qtp1 = Qtable(stp1i,:);
        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
        
        if( ~agentReachedGoal ) 
            delta = rew + gamma * Qp1(actp1) - Q(act);
        else
            delta = rew - Q(act);
        end
        deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
        if grafica,               
             AcrobotPlot(s,ts);
             capture(:,ts) = s';
             F(ts) = getframe;
        end   
        
           if agentReachedGoal,
               if grafica,
                    AcrobotPlot(sp1,ts+1);
               end
               capture(:,ts+1) = sp1';
               F(ts+1) = getframe;
               ts = ts + 1;
               break; 
           end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1;
        error(ts) = max(Qt) - max(Q);
        Q = Qp1;  Qt = Qtp1; 
        if strcmp(functionApproximator,'Qtable'), sti = stp1i; end
        if strcmp(functionApproximator,'regularBPNN'), h = hp1; end
        if strcmp(functionApproximator,'kwtaNN'), id = idp1; h = hp1; end
        ts = ts + 1;
    end % while loop