 function [ Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
% alpha: learning rate
deltaj = (delta * Who(:,act))' .* (1-h) .* h; 
Who(:,act) = Who(:,act) + alpha * delta * h';
biasho(act) = biasho(act) + alpha * delta;
Wih = Wih + alpha * st' * deltaj;
biasih = biasih + alpha * deltaj; 