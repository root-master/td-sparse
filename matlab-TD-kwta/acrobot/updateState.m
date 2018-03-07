function [sp1,agentReachedGoal] = updateState(s,action,omega1_bound,omega2_bound)

torque = action;

theta1 = s(1); omega1 = s(2);
theta2 = s(3); omega2 = s(4);

frequency = 5;
delta_t = 1 / frequency;

% timestep for integration of equations of motion
dt = 0.05;

nSteps_integration = round(delta_t / dt);

%% dynamic and geometric propertiers of links
m1 = 1.0; l1 = 1.0;
m2 = 1.0; l2 = 1.0;

lc1 = l1 / 2; lc2 = l2 / 2;
I1 = 1.0; I2 = 1.0;

g = 9.8;

%% Integration
for i=1:nSteps_integration
    %% see Sutton and Barto's RL book
    d1 = m1*lc1^2 + m2 * (l1^2 + lc2^2 + 2*l1*lc2*cos(theta2)) + I1 + I2;
    d2 = m2 * (lc2^2 + l1*lc2*cos(theta2)) + I2;
    phi2 = m2*lc2 * g * cos(theta1 + theta2 - pi/2);
    phi1 = -m2*l1*lc2 * omega2^2 * sin(theta2) - 2*m2*l1*lc2*omega1*omega2*sin(theta2) + (m1*lc1 + m2*l1) * g * cos(theta1-pi/2) + phi2;

    %% the equation of motion
    alpha2 = (m2*lc2^2 + I2 - d2^2/d1)^(-1) * (torque + d2/d1 * phi1 - m2*l1*lc2*omega1^2 * sin(theta2) - phi2);
    alpha1 = -d1^(-1) * (d2 * alpha2 + phi1);

    %% update angular velocity, (integration)
    omega1 = omega1 + alpha1 * dt;
    % omega1 = 0.999 * omega1;
    omega1 = bound(omega1,omega1_bound);

    omega2 = omega2 + alpha2 * dt;
    % omega2 = 0.999 * omega2;
    omega2 = bound(omega2,omega2_bound);

    %% update angle positions (integration)
    theta1 = theta1 + dt * omega1;
    if (theta1 > pi),
        theta1 = theta1 - 2*pi;
    end
    
    if (theta1 < -pi),
        theta1 = theta1 + 2*pi; 
    end
    
    theta2 = theta2 + dt * omega2;
    if (theta2 < -pi),
        theta2 = theta2 + 2*pi; 
    end

    if (theta2 > pi),
        theta2 = theta2 - 2*pi;
    end
end

y3 = - l1 * cos(theta1) - l2 * cos( theta1 + theta2 ); 
agentReachedGoal = ( y3 >= l1 );

sp1 = [theta1,omega1,theta2,omega2];