function [o,h,id]  = kwta_NN_forward(s, Wih,biasih, Who,biasho) 
    

nCellHidden = length(Wih);

k = round(0.05 * nCellHidden); % number of winners

% propagate Gaussian input state vector to hidden units
net = s * Wih + biasih;

% Sort the net input and find k winners
[netSorted,idsort] = sort(net,'descend');
q = 0.25; % constant 0 < q < 1 determines where exactly
          % to place the inhibition between the k and k + 1th units

biaskwta = netSorted(k+1) + q * ( netSorted(k) - netSorted(k+1) );
id = idsort(1:k);

shunt = 1.0; % shunt is a positive number which is the shift to left in activation-eta
eta = net - biaskwta - shunt; 

% hidden activation

h = 1./(1 + exp(-eta) );

o = h * Who + biasho; % Output