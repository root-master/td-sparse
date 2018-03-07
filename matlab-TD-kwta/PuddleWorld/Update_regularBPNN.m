 function [ Wih,biasih,Who,biasho] = Update_regularBPNN(s,act,h,alpha,delta,Wih,biasih,Who,biasho,withBias)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
deltaj = (delta * Who(:,act))'.* (1-h) .* h;
Who(:,act) = Who(:,act) + alpha * delta * h';
if withBias,
    biasho(act) = biasho(act) + alpha * delta;
else
    biasho = zeros(1,4);
end
Wih = Wih + alpha * s' * deltaj;
%biasih = biasih + alpha * deltaj;
nCellHidden = length(Who);  
biasih = zeros(1,nCellHidden); 