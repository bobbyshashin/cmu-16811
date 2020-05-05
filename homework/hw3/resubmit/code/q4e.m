data = load('cluttered_hallway.txt');
hold on;
scatter3(data(:,1), data(:,2), data(:,3), 'r');
axis([-2 2 -1 1.5 1.5 3])

[x y] = meshgrid(-2:0.1:3);

left = data((data(:,1) < -1.0), :, :);
c_left = ransac(left, 10000, 1e-1)
left_plane = c_left(1)*x + c_left(2)*y + c_left(3);

right = data((data(:,1) > 0.7), :, :);
c_right = ransac(right, 10000, 1e-1)
right_plane = c_right(1)*x + c_right(2)*y + c_right(3);

upper = data((data(:,2) < 0), :, :);
c_upper = ransac(upper, 10000, 1e-1)
upper_plane = c_upper(1)*x + c_upper(2)*y + c_upper(3);

floor = data((data(:,2) > 0.7), :, :);
c_floor = ransac(floor, 10000, 1e-1)
floor_plane = c_floor(1)*x + c_floor(2)*y + c_floor(3);

s_left = calculate_smoothness(left, c_left)
s_right = calculate_smoothness(right, c_right)
s_upper = calculate_smoothness(upper, c_upper)
s_floor = calculate_smoothness(floor, c_floor)

surf(x, y, left_plane, 'FaceColor', 'red')
surf(x, y, right_plane, 'FaceColor', 'green')
surf(x, y, upper_plane, 'FaceColor', 'yellow')
surf(x, y, floor_plane, 'FaceColor', 'blue')

