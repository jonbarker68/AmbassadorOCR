load ocr_data.mat;

data_new = challenge_data;

% Randam noise
data_new(1:20, :) = data_new(1:20, :) + rand(20,900) * 600;

% Inverted
data_new(21:40, :) = 768-data_new(21:40, :);

% Inverted + noise
data_new(41:60, :) = 768-data_new(41:60, :) + rand(20,900) * 300;

% Rotated
data_new(61:80, :) = fliplr(data_new(61:80, :))

% Occluded with white
data_new(81:100, 1:450) = 0

% Occluded with white
data_new(101:120, 1:450) = 750

mask9 = (rand(20,900)<0.9)
mask6 = (rand(20,900)<0.6)
mask3 = (rand(20,900)<0.3)

data_new(121:140, :) = data_new(121:140, :) .* mask9

data_new(141:160, :) = data_new(141:160, :) .* mask6

data_new(161:180, :) = data_new(161:180, :) .* mask3


data_new(181:200, :) = data_new(181:200, :) + 0.5 * data_new(180:199, :)

challenge_data = data_new
clear data_new mask3 mask6 mask9

save ocr_data.mat *
