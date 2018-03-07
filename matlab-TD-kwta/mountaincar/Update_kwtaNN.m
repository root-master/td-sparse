 function [ Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
% alpha: learning rate
deltaj = (- delta * Who(:,act))' .* (1-h) .* h; 
Who(:,act) = Who(:,act) + alpha * delta * h';
% biasho(act) = biasho(act) + alpha * delta;
biasho = zeros(1,length(biasho));
Wih = Wih - alpha * st' * deltaj;
%biasih(id) = biasih(id) - alpha * deltaj(id);
nCellHidden = length(Who);  
biasih = zeros(1,nCellHidden); 