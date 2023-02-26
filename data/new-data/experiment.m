% a = [];
% b = [];
% c = [];
% t = 0;
% for i = [1:10, 20]
%     t = t + 1;
%     n = i * 100000;
%     c = [c, n];
%     s = 'softconstraint\Soft' + string(n) + '.csv';
%     m = readmatrix(s, 'range', [2,1]);
%     a = [a, mean(m(:,5))];
%     b = [b, mean(m(:,6))];
%  %   fprintf("%.5f %.5f %.5f\n",c(t),a(t), b(t));
% end
% 
% 
% figure
% p = plot(c, a, c, b);
% p(1).Marker = 'x';
% p(2).Marker = 'x';
% title('Case 1: \lambda = 0.8');
% ylabel('Regret');
% xlabel('T');
% legend(p,"Algorithm 1", "Baseline");
% grid on
% 


a = [];
b = [];
c = [];
d = [];
t = 0;
for i = [1:10]
    t = t + 1;
    n = i * 100000;
    c = [c, log(n)];
  %  s = 'result_zizhuo\soft\0.2\Soft' + string(n) + '.csv';
  %  s2 = 'result_zizhuo\soft\old\0.2\Soft' + string(n) + '.csv';
    s3 = 'result_zizhuo\soft-new\Soft' + string(n) + '.csv';
   % m = readmatrix(s, 'range', [2,1]);
   % m2 = readmatrix(s2, 'range', [2,1]);
    m3 = readmatrix(s3, 'range', [2,1]);
    a = [a, log(mean(m3(:,5)))];
    b = [b, log(mean(m3(:,6)))];
    d = [d, log(mean(m3(:,7)))];
 %   fprintf("%.5f %.5f %.5f\n",c(t),a(t), b(t));
end

X_1 = [ones(length(c'),1),c']\a'

Y_1 = [ones(length(c'),1),c'] * X_1;

X_2 = [ones(length(c'),1),c']\b'

Y_2 = [ones(length(c'),1),c'] * X_2;

X_3 = [ones(length(c'),1),c']\d'

Y_3 = [ones(length(c'),1),c'] * X_3;
figure
p_1 = scatter(c, a);
p_1.Marker = 'X';
hold on
p_2 = scatter(c, b);
p_2.Marker = 'o';
hold on
p_3 = scatter(c, d);
p_3.Marker = '*';
l = plot(c, Y_1', c, Y_2', c, Y_3');

%p(1).Marker = 'x';
%p(2).Marker = 'x';
title('\lambda = 0.5','Fontsize', 14);
ylabel('log(Regret)','Fontsize', 14);
xlabel('log(T)','Fontsize', 14);
gtext("y = " + string(X_1(2)) + "x - " + string(-X_1(1)));
gtext('y = ' + string(X_2(2)) + 'x - ' + string(-X_2(1)));
gtext('y = ' + string(X_3(2)) + 'x - ' + string(-X_3(1)));
legend("FDP-GFM","Tri-Section","DPA");
grid on

