function best_c = ransac(data, max_iter, threshold)
    max_num_inlier = 0;
    best_c = zeros(3);

    for i = 1:max_iter
        % randomly choose three points in the dataset
        id = randperm(size(data, 1), 3);
        meta = data(id, :);
        
        % calculate the num of inliers with the plane fitted by these three
        % points
        A = meta(:, 1:2);
        b = meta(:, 3);
        A(:, 3) = ones(size(length(b), 1));
        c = inv(A' * A) * A' * b;
        err = abs(A*c - b) / sqrt(c(1)^2 + c(2)^2 + 1);
        num_inlier = sum(err<threshold);
        
        if num_inlier > max_num_inlier
            max_num_inlier = num_inlier;
            best_c = c;
        end
    end

end