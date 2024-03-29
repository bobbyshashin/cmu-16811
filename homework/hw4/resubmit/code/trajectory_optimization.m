%% create obstacle field
close all
clear all
waypoints=300;
N=101;
OBST = [20,30;60,40;70,85];
epsilon = [25; 20; 30];

obs_cost = double(zeros(N));
for i=1:size(OBST,1)

    t = zeros(N);
    t(OBST(i,1),OBST(i,2)) = 1; %point obstacles
    
    t_cost = double(bwdist(t));
    t_cost(t_cost>epsilon(i))=epsilon(i);
    t_cost = 1/(2*epsilon(i))*(t_cost-epsilon(i)).^2;
    
    obs_cost = obs_cost + t_cost(1:N, 1:N);
end

figure(1)
imagesc(obs_cost')
% obstacle cost gradient
gx = diff(double(obs_cost),1,1);
gy = diff(double(obs_cost),1,2);
hold on


figure(1);
surface(1:N,1:N,double(obs_cost'));
xlabel('X')
hold on;

%% initial path
%world params
SX = 10; % START
SY = 10;
GX = 90; % GOAL
GY = 90;


traj = zeros(2,waypoints);
traj(1,1)=SX;
traj(2,1)=SY;
dist_x = GX-SX;
dist_y = GY-SY;
for i=2:waypoints
    traj(1,i)=traj(1,i-1)+dist_x/(waypoints-1);
    traj(2,i)=traj(2,i-1)+dist_y/(waypoints-1);
end

path_init = traj';
tt=size(path_init,1);
path_init_values = zeros(size(path_init,1),1);
for i=1:tt
    path_init_values(i)=obs_cost(floor(path_init(i,1)),floor(path_init(i,2)));
end
plot3(path_init(:,1),path_init(:,2),path_init_values,'.r','MarkerSize',20);
hold on

path = path_init;


%% Optimize it...
% your code comes here
% (a):
% one iteration
% for i=2:tt-1
%     x = fix(path(i,1));
%     y = fix(path(i,2));
%     path(i,1) = path(i,1) - 0.1 * gx(x,y);
%     path(i,2) = path(i,2) - 0.1 * gy(x,y);
% end

% until convergence
% threshold = 1e-5;
% prev_path = path;
% while true
%     for i=2:tt-1
%         x = fix(path(i,1));
%         y = fix(path(i,2));
%         dx = -0.1 * gx(x,y);
%         dy = -0.1 * gy(x,y);
%         path(i,1) = path(i,1) + dx;
%         path(i,2) = path(i,2) + dy;
%     end
%     error = sum((path - prev_path).^2);
%     if error < threshold
%         break;
%     end
%     prev_path = path;
% end

%(b):
% 100, 200, 500
% num_itr = 500;
% threshold = 1e-5;
% prev_path = path;
% for k=1:num_itr
%     for i=2:tt-1
%         x = fix(path(i,1));
%         y = fix(path(i,2));
%         dx = -0.1 * (0.8 * gx(x,y) + 4.0 * (path(i,1) - path(i-1,1)));
%         dy = -0.1 * (0.8 * gy(x,y) + 4.0 * (path(i,2) - path(i-1,2)));
%         path(i,1) = path(i,1) + dx;
%         path(i,2) = path(i,2) + dy;
%     end
%     error = sum((path - prev_path).^2);
%     if error < threshold
%         break;
%     end
%     prev_path = path;
% end

%(c):
% 100, 5000
num_itr = 5000;
threshold = 1e-5;
prev_path = path;
for k=1:num_itr
    for i=2:tt-1
        x = fix(path(i,1));
        y = fix(path(i,2));
        dx = -0.1 * (0.8 * gx(x,y) + 4.0 * (-path(i-1,1) + 2.0 * path(i,1) - path(i+1,1)));
        dy = -0.1 * (0.8 * gy(x,y) + 4.0 * (-path(i-1,2) + 2.0 * path(i,2) - path(i+1,2)));
        path(i,1) = path(i,1) + dx;
        path(i,2) = path(i,2) + dy;
    end
    error = sum((path - prev_path).^2);
    if error < threshold
        break;
    end
    prev_path = path;
end

%% plot the trajectories
path_values = zeros(tt,1);
for i=1:tt
    path_values(i)=obs_cost(floor(path(i,1)),floor(path(i,2)));
end
figure(1)
hold on;
plot3(path(:,1),path(:,2),path_values,'.g','MarkerSize',20);

hold off;
