% Read the image
img = imread('cropped_image.jpg'); % Replace with your image filename

% Convert the image to double precision for processing
img = im2double(img);

% Step 1: Visualize the image to identify green edges (for debugging)
figure;
imshow(img);
title('Original Image');

% Step 2: Detect the green edges dynamically
% Extract the RGB channels
redChannel = img(:,:,1);
greenChannel = img(:,:,2);
blueChannel = img(:,:,3);

% Create a mask for green pixels (dynamic thresholding)
% Green edges usually have higher green intensity compared to red and blue
greenMask = (greenChannel > 1.2 * redChannel) & (greenChannel > 1.2 * blueChannel);

% Step 3: Visualize the green mask for debugging
figure;
imshow(greenMask);
title('Green Edge Mask (Dynamic Detection)');
pause(5); % Pause to inspect the mask before proceeding

% Step 4: Remove the green edge by inpainting
% Inpaint the green edge region for each channel
restoredImg = img; % Initialize the restored image
for channel = 1:3
    % Extract the current channel
    currentChannel = img(:,:,channel);
    
    % Apply regionfill to the current channel using the greenMask
    restoredImg(:,:,channel) = regionfill(currentChannel, greenMask);
end

% Step 5: Perform edge-preserving smoothing using guided filtering
radius = 8; % Radius of the filter
epsilon = 0.01; % Regularization parameter for edge preservation

% Apply guided filtering to balance noise reduction and edge preservation
smoothedImg = imguidedfilter(restoredImg, 'NeighborhoodSize', [radius radius], 'DegreeOfSmoothing', epsilon);

% Step 6: Display the results
figure;
subplot(1, 3, 1);
imshow(img);
title('Original Image');

subplot(1, 3, 2);
imshow(restoredImg);
title('After Inpainting');

subplot(1, 3, 3);
imshow(smoothedImg);
title('Final Restored Image');

% Save the final output
imwrite(smoothedImg, 'restored_image.jpg');
