function [o,sh,gh,h]  = kwta_NN_forward_2chunk(st,gt,Wsh,Wgh,bsh,bgh,Wsgh,bsgh,Who,bho) 

shunt = 1;

nsh = length(Wsh);
ngh = length(Wgh);

k1 = 0.1;
ksh = round(k1* nsh); % number of winners
kgh = round(k1* ngh);

net_sh = st * Wsh + bsh;
net_gh = gt * Wgh + bgh;

[netSorted,idsort] = sort(net_sh,'descend');
q = 0.25; % constant 0 < q < 1 determines where exactly
          % to place the inhibition between the k and k + 1th units

bias_kwta_sh = netSorted(ksh+1) + q * ( netSorted(ksh) - netSorted(ksh+1) );
id_sh = idsort(1:ksh);

[netSorted,idsort] = sort(net_gh,'descend');
q = 0.25; % constant 0 < q < 1 determines where exactly
          % to place the inhibition between the k and k + 1th units

bias_kwta_gh = netSorted(kgh+1) + q * ( netSorted(kgh) - netSorted(kgh+1) );
id_gh = idsort(1:kgh);

eta_sh = net_sh - bias_kwta_sh - shunt; % shunt is a positive number which is the shift to left in activation-eta
eta_gh = net_gh - bias_kwta_gh - shunt;

sh = eta_sh;
gh = eta_gh;

eta_sgh = [eta_sh, eta_gh];
h_sgh = 1./(1 + exp(-eta_sgh) );

% h_sh = zeros(size(eta_sh));
% h_sh(id_sh) = 1./(1 + exp(-eta_sh(id_sh)) );
% 
% h_gh = zeros(size(eta_gh));
% h_gh(id_gh) = 1./(1 + exp(-eta_gh(id_gh)) );
% 
% h_sgh = [h_sh,h_gh];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k2 = 0.1;
nh = length(Who);
kh = round(k2*nh);

net_h = h_sgh * Wsgh + bsgh;
[netSorted,idsort] = sort(net_h,'descend');
q = 0.25; % constant 0 < q < 1 determines where exactly
          % to place the inhibition between the k and k + 1th units

bias_kwta_h = netSorted(kh+1) + q * ( netSorted(kh) - netSorted(kh+1) );
id_h = idsort(1:kgh);

eta_h = net_h - bias_kwta_h - shunt;
h = 1./(1 + exp(-eta_h) );

% h = zeros(size(eta_h));
% h(id_h) = 1./(1 + exp(-eta_h(id_h)) );
o = h * Who + bho; % Output




