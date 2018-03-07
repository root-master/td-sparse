 function [ Wih,biasih,Who,biasho] = Update_regularBPNN(s,act,h,alpha,delta,Wih,biasih,Who,biasho)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
deltaj = (delta * Who(:,act))'.* (1-h) .* h;
Who(:,act) = Who(:,act) + alpha * delta * h';

%    biasho(act) = biasho(act) + alpha * delta;
biasho = zeros(1,length(biasho));
Wih = Wih + alpha * s' * deltaj;
%biasih = biasih + alpha * deltaj;
nCellHidden = length(Who);  
biasih = zeros(1,nCellHidden); 