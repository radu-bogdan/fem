% Modellparameter
ell=2; kappa=1; gamma=1;
f=@(x) (x>=0.3 & x<=0.6)*1.0;
% Gitter
N = 10;
dz=ell/N; zk=[0:dz:ell]; nz=length(zk);
% Assemblierung
A=sparse(nz,nz); b=zeros(nz,1);
for k=1:N
hk=zk(k+1)-zk(k);
Kk=kappa/hk*[1,-1; -1,1];
Mk=gamma*hk/6*[2,1;1,2];
% Berechnung von bk mit Mittelpunktsregel:
% Integrationsstellen und Gewichte
zl=[(zk(k)+zk(k+1))/2]; wl=1;
% Werte der Basisfunktionen
phi1=[0.5]; phi2=[0.5];
bk=[ hk*sum(wl.*f(zl).*phi1);
hk*sum(wl.*f(zl).*phi2)];
% Aufsummieren aller Eintr ̈age
ii=[k,k+1]; jj=[k,k+1];
A(jj,ii)=A(ii,jj)+Kk+Mk;
b(ii) =b(ii) +bk;
end
% L ̈osen des Problems
uh = A\b;
% Plotten des Ergebnisses
plot(zk,uh);