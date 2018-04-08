 function [ Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,h_id,alpha,delta,Wih,biasih,Who,biasho)
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

% deltaj = (error * Who(:,act))' .* (1-h) .* h; 
% Who(:,act) = Who(:,act) - alpha * error * h';
% biasho(act) = biasho(act) - alpha * error;
% Wih = Wih - alpha * st' * deltaj;
% biasih = biasih - alpha * deltaj;

% deltaj = zeros(1,length(h));
% deltaj(h_id) = (- delta * Who(h_id,act))' .* (1-h(h_id)) .* h((h_id)); 
% Who(h_id,act) = Who(h_id,act) + alpha * delta * h(h_id)';
% biasho(act) = biasho(act) + alpha * delta;
% Wih(:,h_id) = Wih(:,h_id) - alpha * st' * deltaj(h_id);
% biasih(h_id) = biasih(h_id) - alpha * deltaj(h_id);

st_id = st > 0.05;
deltaj = zeros(1,length(h));
deltaj(h_id) = (- delta * Who(h_id,act))' .* (1-h(h_id)) .* h((h_id)); 
Who(h_id,act) = Who(h_id,act) + alpha * delta * h(h_id)';
biasho(act) = biasho(act) + alpha * delta;
Wih(st_id,h_id) = Wih(st_id,h_id) - alpha * st(st_id)' * deltaj(h_id);
biasih(h_id) = biasih(h_id) - alpha * deltaj(h_id);



