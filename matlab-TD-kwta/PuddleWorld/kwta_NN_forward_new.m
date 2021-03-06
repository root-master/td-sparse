function [o,h,id]  = kwta_NN_forward_new(s, Wih,biasih, Who,biasho) 

shunt = 1;

nCellHidden = length(Wih);

k_rate = 0.1;

k = round(k_rate* nCellHidden); % number of winners

% net = zeros(nCellHidden,1);

% forward pass
% propagate input to hidden
net = s * Wih + biasih;

[netSorted,idsort] = sort(net,'descend');
q = 0.25; % constant 0 < q < 1 determines where exactly
          % to place the inhibition between the k and k + 1th units

biaskwta = netSorted(k+1) + q * ( netSorted(k) - netSorted(k+1) );
id = idsort(1:k);

eta = net - biaskwta - shunt; % shunt is a positive number which is the shift to left in activation-eta

% hidden activation
% h = zeros(size(eta));
% h(id) = 1./(1 + exp(-eta(id)) );
h = 1./(1 + exp(-eta) );

o = h * Who + biasho; % Output

