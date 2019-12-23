%% CS510 -  Introduction to Artificial Intelligence Final Project

% Use of AlexNet to classify images
%   need deep learning toolbox
%   need AlexNet toolbox

% Script for creating adversarial images to fool machine learning
net = inceptionv3();
sz = net.Layers(1).InputSize;
classNames = net.Layers(end).ClassNames;

dir_orig = 'Images/Original_Examples/';
dir_adv = 'Images/Adversarial_Examples/';
dir_rob_0 = 'Images/Robust_Adversarial/';
dir_rob_100 = 'Images/Robust_Adversarial_100/';

origFiles = dir(fullfile(dir_orig,'*.jpg*'));
advFiles = dir(fullfile(dir_adv,'*.png*'));
robFiles = dir(fullfile(dir_rob_0,'*.png*'));
rob100Files = dir(fullfile(dir_rob_100,'*.png*'));

for k = 1:22
    %old_img = imread([char(origFiles(k).folder),'/',char(origFiles(k).name)]);
    %adv_img = imread([char(advFiles(k).folder),'/',char(advFiles(k).name)]);
    rob0_img = imread([char(robFiles(k).folder),'/',char(robFiles(k).name)]);
    rob1_img = imread([char(rob100Files(k).folder),'/',char(rob100Files(k).name)]);
    oldcrop_img = imresize(old_img,[sz(1),sz(2)]);
    %oldcrop_img = imgaussfilt(adv_img,1);
    %img_denoised = imresize(wavelet_denoise(adv_img,3),[sz(1),sz(2)]);
    img_denoisedr0 = imresize(wavelet_denoise(rob0_img,10),[sz(1),sz(2)]);
    img_denoisedr1 = imresize(wavelet_denoise(rob1_img,10),[sz(1),sz(2)]);
    %[orig_label,scores_i] = classify(net,oldcrop_img);
    %[adv_label,scores_a] = classify(net,adv_img);
    %[denoised_label,scores_o] = classify(net,img_denoised);
    [denoisedr0_label,scores_r0o] = classify(net,img_denoisedr0);
    [denoisedr1_label,scores_r1o] = classify(net,img_denoisedr1);
    
    denoisedr0_label
    denoisedr1_label

%     figure;
    %subplot(1,5,1);
    %imagesc(oldcrop_img); axis off;
    %title([string(orig_label)+ ', '+ num2str(100*scores_i(classNames == orig_label))+ '%']);
    %subplot(1,5,2);
    %imagesc(adv_img); axis off;
    %title([string(adv_label)+ ', '+ num2str(100*scores_a(classNames == adv_label),3)+ '%']);
    %subplot(1,5,3);
    %imagesc(img_denoised); axis off;
    %title([string(denoised_label)+ ', '+ num2str(100*scores_o(classNames == denoised_label),3)+ '%']);
%         subplot(1,2,1);
%     imagesc(img_denoisedr0); axis off;
%     title([string(denoisedr0_label)+ ', '+ num2str(100*scores_r0o(classNames == denoisedr0_label),3)+ '%']);
%         subplot(1,2,2);
%     imagesc(img_denoisedr1); axis off;
%     title([string(denoisedr1_label)+ ', '+ num2str(100*scores_r1o(classNames == denoisedr1_label),3)+ '%']);
end

%% Gaussian vs. Wavelet

 k = 13;
    old_img = imread([char(origFiles(k).folder),'/',char(origFiles(k).name)]);
    adv_img = imread([char(advFiles(k).folder),'/',char(advFiles(k).name)]);
    rob0_img = imread([char(robFiles(k).folder),'/',char(robFiles(k).name)]);
    rob1_img = imread([char(rob100Files(k).folder),'/',char(rob100Files(k).name)]);
    oldcrop_img = imresize(old_img,[sz(1),sz(2)]);
    gauss_img = imgaussfilt(adv_img,1.46);
    img_denoised = imresize(wavelet_denoise(adv_img),[sz(1),sz(2)]);
    [orig_label,scores_i] = classify(net,adv_img);
    [adv_label,scores_a] = classify(net,gauss_img);
    [denoised_label,scores_o] = classify(net,img_denoised);

    figure;
    subplot(1,3,1);
    imagesc(oldcrop_img); axis off;
    title([string(orig_label)+ ', '+ num2str(100*scores_i(classNames == orig_label))+ '%']);
    subplot(1,3,2);
    imagesc(gauss_img); axis off;
    title([string(adv_label)+ ', '+ num2str(100*scores_a(classNames == adv_label),3)+ '%']);
    subplot(1,3,3);
    imagesc(img_denoised); axis off;
    title([string(denoised_label)+ ', '+ num2str(100*scores_o(classNames == denoised_label),3)+ '%']);

%% Plot

% Use of AlexNet to classify images
%   need deep learning toolbox
%   need AlexNet toolbox

% Script for creating adversarial images to fool machine learning
net = inceptionv3();
sz = net.Layers(1).InputSize;
classNames = net.Layers(end).ClassNames;

dir_orig = 'Images/Original_Examples/';
dir_adv = 'Images/Robust_Adversarial/';

origFiles = dir(fullfile(dir_orig,'*.jpg*'));
advFiles = dir(fullfile(dir_adv,'*.png*'));
net_acc = zeros([3 10]);

for k=1:length(advFiles)
    for j=1:10
    old_img = imread([char(origFiles(k).folder),'/','panda.jpg']);
    adv_img = imread([char(advFiles(k).folder),'/',char(advFiles(k).name)]);
    rob0_img = imread([char(robFiles(k).folder),'/',char(robFiles(k).name)]);
    rob1_img = imread([char(rob100Files(k).folder),'/',char(rob100Files(k).name)]);
    img_denoised = imresize(wavelet_denoise(adv_img,j),[sz(1),sz(2)]);
    old_img = imresize(old_img,[sz(1),sz(2)]);
    [orig_label,scores_i] = classify(net,old_img);
    [adv_label,scores_a] = classify(net,adv_img);
    [denoised_label,scores_o] = classify(net,img_denoised);
    net_acc(k,j) = 100*scores_o(classNames == denoised_label);
    if denoised_label == adv_label
        net_acc(k,j) = -net_acc(k,j);
    end
    end
end

figure;
plot(net_acc','LineWidth',2); grid on; hold on;
plot([1 2 3 4 5 6 7 8 9 10],zeros([1 10]),'--k', 'LineWidth',2);
ylim([-100 100]); 
xlabel('Wavelet Sparsity Parameter'); ylabel('Classification Accuracy (%)');
title('Inception v3 Accuracy (Giant Panda)');
legend('Invariant, \epsilon - 0.03', 'RI, \epsilon = 0.03', 'RI, \epsilon = 1');

    