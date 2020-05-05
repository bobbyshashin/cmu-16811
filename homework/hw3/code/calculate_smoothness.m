function smoothness = calculate_smoothness(data, c)

A = data(:, 1:2);
b = data(:, 3);
A(:, 3) = ones(size(length(b), 1));

err = abs(A*c - b) / sqrt(c(1)^2 + c(2)^2 + 1);
smoothness = sum(err) / length(err);

end