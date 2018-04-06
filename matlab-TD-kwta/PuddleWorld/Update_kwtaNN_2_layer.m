 function [ Wih,biasih,Wh1h2,biash1h2,Who,biasho] = Update_kwtaNN_2_layer(st,act,h_1,h_1_id,h_2,h_2_id,alpha,delta,Wih,biasih, Wh1h2,biash1h2, Who,biasho)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
% alpha: learning rate

error = - delta;

error_vec = zeros(1,4);
error_vec(act) = error;
% 
deltaj = (error_vec * Who') .* (1-h_2) .* h_2; 
Who = Who - alpha * h_2' * error_vec;
biasho = biasho - alpha * error_vec;

deltaj_1 = (deltaj * Wh1h2') .* (1-h_1) .* h_1; 
Wh1h2 = Wh1h2 - alpha * h_1' * deltaj;
biash1h2 = biash1h2 - alpha * deltaj;


Wih = Wih - alpha * st' * deltaj_1;
biasih = biasih - alpha * deltaj_1;


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

% st_id = st > 0.1;
% deltaj = zeros(1,length(h));
% deltaj(h_id) = (- delta * Who(h_id,act))' .* (1-h(h_id)) .* h((h_id)); 
% Who(h_id,act) = Who(h_id,act) + alpha * delta * h(h_id)';
% biasho(act) = biasho(act) + alpha * delta;
% Wih(st_id,h_id) = Wih(st_id,h_id) - alpha * st(st_id)' * deltaj(h_id);
% biasih(h_id) = biasih(h_id) - alpha * deltaj(h_id);



