 function [Wsh,Wgh,bsh,bgh,Wsgh,bsgh, Who,bho] = Update_kwtaNN_2_chunk(st,gt,act,sh,gh,h,alpha,delta,Wsh,Wgh,bsh,bgh,Wsgh,bsgh, Who,bho)

error = - delta;

deltaj_2 = (error * Who(:,act))' .* (1-h) .* h; 
Who(:,act) = Who(:,act) - alpha * error * h';
bho(act) = bho(act) - alpha * error;

h_1 = [sh,gh];
deltaj_1 = (deltaj_2 * Wsgh') .* (1-h_1) .* h_1; 
Wsgh = Wsgh - alpha * h_1' * deltaj_2;
bsgh = bsgh - alpha * deltaj_2;

deltaj_s = deltaj_1(1:length(Wsh));
deltaj_g = deltaj_1(length(Wsh)+1:end);
Wsh = Wsh - alpha * st' * deltaj_s;
Wgh = Wgh - alpha * gt' * deltaj_g;

bsh = bsh - alpha * deltaj_s;
bgh = bgh - alpha * deltaj_g;
