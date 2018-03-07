function [Wio,biasio] = UpdateSimpleNN(st,act,alpha,delta,Wio,biasio)

Wio(:,act) = Wio(:,act) + alpha * delta * st';
biasio(act) = biasio(act) + alpha * delta;