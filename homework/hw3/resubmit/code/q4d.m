data = load('clean_hallway.txt');
hold on;
scatter3(data(:,1), data(:,2), data(:,3), 'r');
axis([-1 3 -1 3 1 3]);

[x y] = meshgrid(-1:0.1:3);

left = data((data(:,1) < 0.5), :, :);
c_left = ransac(left, 10000, 1e-3)
left_plane = c_left(1)*x + c_left(2)*y + c_left(3);

right = data((data(:,1) > 2.0), :, :);
c_right = ransac(right, 10000, 1e-3)
right_plane = c_right(1)*x + c_right(2)*y + c_right(3);

upper = data((data(:,2) < 0.5), :, :);
c_upper = ransac(upper, 10000, 1e-3)
upper_plane = c_upper(1)*x + c_upper(2)*y + c_upper(3);

floor = data((data(:,2) > 2.0), :, :);
c_floor = ransac(floor, 10000, 1e-3)
floor_plane = c_floor(1)*x + c_floor(2)*y + c_floor(3);

surf(x, y, left_plane, 'FaceColor', 'red')
surf(x, y, right_plane, 'FaceColor', 'green')
surf(x, y, upper_plane, 'FaceColor', 'yellow')
surf(x, y, floor_plane, 'FaceColor', 'blue')
