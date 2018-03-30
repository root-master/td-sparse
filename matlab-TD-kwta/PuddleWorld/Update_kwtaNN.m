 function [ Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,alpha,delta,Wih,biasih,Who,biasho)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
% alpha: learning rate

error = - delta;

error_vec = zeros(1,4);
error_vec(act) = error;

deltaj = (error_vec * Who') .* (1-h) .* h; 
Who = Who - alpha * h' * error_vec;
biasho = biasho - alpha * error_vec;
Wih = Wih - alpha * st' * deltaj;
biasih = biasih - alpha * deltaj;

% deltaj = (error * Who(:,act))' .* (1-h) .* h; 
% Who(:,act) = Who(:,act) - alpha * error * h';
% biasho(act) = biasho(act) - alpha * error;
% Wih = Wih - alpha * st' * deltaj;
% biasih = biasih - alpha * deltaj;

