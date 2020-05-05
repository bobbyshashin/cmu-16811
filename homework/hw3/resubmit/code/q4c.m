data = load('cluttered_table.txt');
hold on;
scatter3(data(:,1), data(:,2), data(:,3), 'r');

% A = data(:, 1:2);
% A(:, 3) = ones(size(length(b), 1));
% b = data(:, 3);
% c = inv(A' * A) * A' * b
c = ransac(data, 10000, 1e-3)
[x y] = meshgrid(-1.5:3:3);
z = c(1)*x + c(2)*y + c(3);
% mesh(x,y,z)

axis([-1.5 1.5 0 0.55 1.5 2.5]);
surf(x,y,z)

err = abs(A*c - b) / sqrt(c(1)^2 + c(2)^2 + 1);
avg_dist = sum(err) / length(err)