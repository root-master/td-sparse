 function [ Wih,biasih,Who,biasho] = Update_kwtaNN(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias)
% Update_kwtaNN update the weigths of kwta neural net
% st: previous state before taking action (act)
% Q : output for st
% alpha: learning rate
deltaj = zeros(1,length(h));
deltaj(id) = (- delta * Who(id,act))' .* (1-h(id)) .* h((id)); 
Who(id,act) = Who(id,act) + alpha * delta * h(id)';
if withBias,
    biasho(act) = biasho(act) + alpha * delta;
else
    biasho = zeros(1,4);
end
Wih(:,id) = Wih(:,id) - alpha * st' * deltaj(id);
%biasih(id) = biasih(id) - alpha * deltaj(id);
nCellHidden = length(Who);  
biasih = zeros(1,nCellHidden); 