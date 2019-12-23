function img_w = wavelet_denoise(img,j)

min_i = double(min(min(min(img,[],3))));
max_i = double(max(max(max(img,[],3))));
for d=1:size(img,3)
    img_i(:,:,d) = uint8(255*(double(img(:,:,d)) - min_i)./(max_i-min_i));
end

wname = 'bior3.5';
level = 5; sorh = 's';
for d=1:size(img_i,3)
    [C,S] = wavedec2(img_i(:,:,d),level,wname);
    thr = wthrmngr('dw2ddenoLVL','penalhi',C,S,j);
    [out,~,~] = wdencmp('lvd',C,S,wname,level,thr,sorh);
    img_w(:,:,d) = uint8(out);
end

%{
figure;
subplot(1,2,1);
imagesc(img_i); axis off;
title('Noisy Image');
subplot(1,2,2);
imagesc(img_w); axis off;
title('Denoised Image');
%}