% myDir = './Images/Original_Examples'; %uigetdir; %gets directory
myDir = './Images/Adversarial_Examples'; %uigetdir; %gets directory
% myDir = './Images/new';

% myFiles = dir(fullfile(myDir,'*.j*')); %gets all wav files in struct
myFiles = dir(fullfile(myDir,'*.png*')); %gets all wav files in struct

net = inceptionv3();
sz = net.Layers(1).InputSize;

for k = 1:length(myFiles)
  img = imread([char(myFiles(k).folder),'/',char(myFiles(k).name)]);
  img = imresize(img,[sz(1),sz(2)]);
  label = string(classify(net,img));
  
  figure;
  imagesc(img)
  title(label)
end