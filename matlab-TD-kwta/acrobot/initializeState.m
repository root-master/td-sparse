function s = initializeState(theta1_vec,omega1_vec,theta2_vec,omega2_vec)
% random state (x,y) inside gridworld
theta1 = theta1_vec ( randi( length(theta1_vec) ) );
theta2 = theta2_vec ( randi( length(theta2_vec) ) );
omega1 = omega1_vec ( randi( length(omega1_vec) ) );
omega2 = omega2_vec ( randi( length(omega1_vec) ) );

s = [theta1,omega1,theta2,omega2];