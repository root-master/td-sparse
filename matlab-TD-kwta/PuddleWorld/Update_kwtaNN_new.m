 function [ Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
% alpha: learning rate
deltaj = (- delta * Who(:,act))' .* (1-h) .* h; 
Who(:,act) = Who(:,act) + alpha * delta * h';
if withBias,
    biasho(act) = biasho(act) + alpha * delta;
else
    biasho = zeros(1,4);
end
Wih = Wih - alpha * st' * deltaj;
%biasih(id) = biasih(id) - alpha * deltaj(id);
nCellHidden = length(Who);  
biasih = zeros(1,nCellHidden); 