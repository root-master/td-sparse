 function [ Wih,biasih,Wh1h2,biash1h2,Who,biasho] = Update_kwtaNN_2_layer(st,act,h_1,h_1_id,h_2,h_2_id,alpha,delta,Wih,biasih, Wh1h2,biash1h2, Who,biasho)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
% alpha: learning rate

error = - delta;
% 
% error_vec = zeros(1,4);
% error_vec(act) = error;
% % 
% deltaj = (error_vec * Who') .* (1-h_2) .* h_2; 
% Who = Who - alpha * h_2' * error_vec;
% biasho = biasho - alpha * error_vec;
% 
% deltaj_1 = (deltaj * Wh1h2') .* (1-h_1) .* h_1; 
% Wh1h2 = Wh1h2 - alpha * h_1' * deltaj;
% biash1h2 = biash1h2 - alpha * deltaj;
% 
% 
% Wih = Wih - alpha * st' * deltaj_1;
% biasih = biasih - alpha * deltaj_1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RADICAL METHOD %
% Only update winners %
deltaj_2 = (error * Who(h_2_id,act))' .* (1-h_2(h_2_id)) .* h_2(h_2_id); 
Who(h_2_id,act) = Who(h_2_id,act) - alpha * error * h_2(h_2_id)';
biasho(act) = biasho(act) - alpha * error;

deltaj_1 = (deltaj_2 * Wh1h2(h_1_id,h_2_id)') .* (1-h_1(h_1_id)) .* h_1(h_1_id); 
Wh1h2(h_1_id,h_2_id) = Wh1h2(h_1_id,h_2_id) - alpha * h_1(h_1_id)' * deltaj_2;
biash1h2(h_2_id) = biash1h2(h_2_id) - alpha * deltaj_2;

st_id = st > 0.1;
Wih(st_id,h_1_id) = Wih(st_id,h_1_id) - alpha * st(st_id)' * deltaj_1;
biasih(h_1_id) = biasih(h_1_id) - alpha * deltaj_1;


