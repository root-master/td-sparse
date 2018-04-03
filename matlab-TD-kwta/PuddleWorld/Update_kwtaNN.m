 function [ Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
% alpha: learning rate

error = - delta;

% error_vec = zeros(1,4);
% error_vec(act) = error;
% 
% deltaj = (error_vec * Who') .* (1-h) .* h; 
% Who = Who - alpha * h' * error_vec;
% biasho = biasho - alpha * error_vec;
% Wih = Wih - alpha * st' * deltaj;
% biasih = biasih - alpha * deltaj;

deltaj = (error * Who(:,act))' .* (1-h) .* h; 
Who(:,act) = Who(:,act) - alpha * error * h';
biasho(act) = biasho(act) - alpha * error;
Wih = Wih - alpha * st' * deltaj;
biasih = biasih - alpha * deltaj;

% deltaj = zeros(1,length(h));
% deltaj(id) = (- delta * Who(id,act))' .* (1-h(id)) .* h((id)); 
% Who(id,act) = Who(id,act) + alpha * delta * h(id)';
% biasho(act) = biasho(act) + alpha * delta;
% Wih(:,id) = Wih(:,id) - alpha * st' * deltaj(id);
% biasih(id) = biasih(id) - alpha * deltaj(id);