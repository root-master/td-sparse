function [Q,h]  = regularBPNN_forward(s, Wih,biasih, Who,biasho) 
    
% forward pass
% propagate input to hidden
eta = s * Wih + biasih;

% hidden activation
h = 1./(1+exp(-eta));

Q = h * Who + biasho; % Output
