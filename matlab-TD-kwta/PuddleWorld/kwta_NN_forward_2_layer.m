function [o,h_1,h_1_id,h_2,h_2_id]  = kwta_NN_forward_2_layer(s, Wih,biasih, Wh1h2,biash1h2, Who,biasho) 

shunt = 1;

nCellHidden = length(Wih);

k1_rate = 0.1;
k2_rate = 0.2;

k = round(k1_rate * nCellHidden); % number of winners

% net = zeros(nCellHidden,1);

%%%%%% LAYER 1 %%%%%%%%%%%
net = s * Wih + biasih;

[netSorted,idsort] = sort(net,'descend');
q = 0.25; % constant 0 < q < 1 determines where exactly
          % to place the inhibition between the k and k + 1th units

biaskwta = netSorted(k+1) + q * ( netSorted(k) - netSorted(k+1) );
h_1_id = idsort(1:k);

eta = net - biaskwta - shunt; % shunt is a positive number which is the shift to left in activation-eta

% hidden activation
h_1 = zeros(size(eta));
h_1(h_1_id) = 1./(1 + exp(-eta(h_1_id)) );
%h_1 = 1./(1 + exp(-eta) );

%%%%%%%% LAYER 2 %%%%%%%

nCellHidden2 = length(Wh1h2);
k = round(k2_rate * nCellHidden2); % number of winners

net = h_1 * Wh1h2 + biash1h2;
[netSorted,idsort] = sort(net,'descend');
q = 0.25; % constant 0 < q < 1 determines where exactly
          % to place the inhibition between the k and k + 1th units

biaskwta = netSorted(k+1) + q * ( netSorted(k) - netSorted(k+1) );
h_2_id = idsort(1:k);

eta = net - biaskwta - shunt; % shunt is a positive number which is the shift to left in activation-eta

% hidden activation
h_2 = zeros(size(eta));
h_2(h_2_id) = 1./(1 + exp(-eta(h_2_id)) );

%h_2 = 1./(1 + exp(-eta) );

o = h_2 * Who + biasho; % Output




