% uses MatLab to recognize features between images and match side by side pictures and combine them into 1 panorama type photo
% By kyle granville

% phone camera aspect ratio is about 750:355
% so images have been resized to that

clc
clear
close all

% reading images
im11 = imread('S1-im1a.png');
im12 = imread('S1-im2a.png');
im13 = imread('S1-im3a.png');
im21 = imread('S2-im1a.png');
im22 = imread('S2-im2a.png');
im23 = imread('S2-im3a.png');
im24 = imread('S2-im4a.png');
im25 = imread('S2-im5a.png');
im26 = imread('S2-im6a.png');
im27 = imread('S2-im7a.png');
im31 = imread('S3-im1a.png');
im32 = imread('S3-im2a.png');
im33 = imread('S3-im3a.png');
im34 = imread('S3-im4a.png');
im41 = imread('S4-im1a.png');
im42 = imread('S4-im2a.png');

%image resizing
im11 = imresize(im11, [355 750]);
im12 = imresize(im12, [355 750]);
im13 = imresize(im13, [355 750]);
im21 = imresize(im21, [750 350]);
im22 = imresize(im22, [750 350]);
im23 = imresize(im23, [750 350]);
im24 = imresize(im24, [750 350]);
im25 = imresize(im25, [750 350]);
im26 = imresize(im26, [750 350]);
im27 = imresize(im27, [750 350]);
im31 = imresize(im31, [750 350]);
im32 = imresize(im32, [750 350]);
im33 = imresize(im33, [750 350]);
im34 = imresize(im34, [750 350]);
im41 = imresize(im41, [355 750]);
im42 = imresize(im42, [355 750]);

%saving resized images
imwrite(im11,'S1-im1.png');
imwrite(im12,'S1-im2.png');
imwrite(im13,'S1-im3.png');
imwrite(im21,'S2-im1.png');
imwrite(im22,'S2-im2.png');
imwrite(im23,'S2-im3.png');
imwrite(im24,'S2-im4.png');
imwrite(im25,'S2-im5.png');
imwrite(im26,'S2-im6.png');
imwrite(im27,'S2-im7.png');
imwrite(im31,'S3-im1.png');
imwrite(im32,'S3-im2.png');
imwrite(im33,'S3-im3.png');
imwrite(im34,'S3-im4.png');
imwrite(im41,'S4-im1.png');
imwrite(im42,'S4-im2.png');

%image preparation (rgb2gray)
im11 = im2double(im11);
im12 = im2double(im12);
im13 = im2double(im13);
im11g = rgb2gray(im11);
im12g = rgb2gray(im12);
im13g = rgb2gray(im13);
im21 = im2double(im21);
im22 = im2double(im22);
im23 = im2double(im23);
im24 = im2double(im24);
im25 = im2double(im25);
im26 = im2double(im26);
im27 = im2double(im27);
im21g = rgb2gray(im21);
im22g = rgb2gray(im22);
im23g = rgb2gray(im23);
im24g = rgb2gray(im24);
im25g = rgb2gray(im25);
im26g = rgb2gray(im26);
im27g = rgb2gray(im27);
im31 = im2double(im31);
im32 = im2double(im32);
im33 = im2double(im33);
im34 = im2double(im34);
im31g = rgb2gray(im31);
im32g = rgb2gray(im32);
im33g = rgb2gray(im33);
im34g = rgb2gray(im34);
im41 = im2double(im41);
im42 = im2double(im42);
im41g = rgb2gray(im41);
im42g = rgb2gray(im42);

%saving into arrays for panorama cropping later
im1 = zeros([size(im11),3]);
im1(:,:,:,1) = im11;
im1(:,:,:,2) = im12;
im1(:,:,:,3) = im13;
im1g = [im11g, im12g, im13g];
im2 = zeros([size(im21),7]);
im2g = zeros([size(im21),7]);
im2(:,:,:,1) = im21;
im2(:,:,:,2) = im22;
im2(:,:,:,3) = im23;
im2(:,:,:,4) = im24;
im2(:,:,:,5) = im25;
im2(:,:,:,6) = im26;
im2(:,:,:,7) = im27;
im2g(:,:,1) = im21g;
im2g(:,:,2) = im22g;
im2g(:,:,3) = im23g;
im2g(:,:,4) = im24g;
im2g(:,:,5) = im25g;
im2g(:,:,6) = im26g;
im2g(:,:,7) = im27g;
im3 = zeros([size(im31),4]);
im3(:,:,:,1) = im31;
im3(:,:,:,2) = im32;
im3(:,:,:,3) = im33;
im3(:,:,:,4) = im34;
im3g = zeros([size(im31),4]);
im3g(:,:,1) = rgb2gray(im31);
im3g(:,:,2) = rgb2gray(im32);
im3g(:,:,3) = rgb2gray(im33);
im3g(:,:,4) = rgb2gray(im34);
im4 = zeros([size(im41),2]);
im4(:,:,:,1) = im41;
im4(:,:,:,2) = im42;
im4g = zeros([size(im41),2]);
im4g(:,:,1) = im41g;
im4g(:,:,2) = im42g;
%imshow(im1); figure;
%imshow(im2);figure;
%imshow(im3);figure;

%filter kernels
gausfilt = fspecial('gaussian',21,3);
dog = [-1 0 -1; -2 0 2; 1 0 1];
%

%filtering images
im11filt = imfilter(im11g,gausfilt);
im12filt = imfilter(im12g,gausfilt);
im13filt = imfilter(im13g,gausfilt);
im21filt = imfilter(im21g,gausfilt);
im22filt = imfilter(im22g,gausfilt);
im23filt = imfilter(im23g,gausfilt);
im24filt = imfilter(im24g,gausfilt);
im25filt = imfilter(im25g,gausfilt);
im26filt = imfilter(im26g,gausfilt);
im27filt = imfilter(im27g,gausfilt);
im31filt = imfilter(im31g,gausfilt);
im32filt = imfilter(im32g,gausfilt);
im33filt = imfilter(im33g,gausfilt);
im34filt = imfilter(im34g,gausfilt);
im41filt = imfilter(im41g,gausfilt);
im42filt = imfilter(im42g,gausfilt);

% harris corner detection scores, from lecture method
im11cor = harcor(im11g,dog,gausfilt);
im12cor = harcor(im12g,dog,gausfilt);
im13cor = harcor(im13g,dog,gausfilt);
im21cor = harcor(im21g,dog,gausfilt);
im22cor = harcor(im22g,dog,gausfilt);
im23cor = harcor(im23g,dog,gausfilt);
im24cor = harcor(im24g,dog,gausfilt);
im25cor = harcor(im25g,dog,gausfilt);
im26cor = harcor(im26g,dog,gausfilt);
im27cor = harcor(im27g,dog,gausfilt);
im31cor = harcor(im31g,dog,gausfilt);
im32cor = harcor(im32g,dog,gausfilt);
im33cor = harcor(im33g,dog,gausfilt);
im34cor = harcor(im34g,dog,gausfilt);
im41cor = harcor(im41g,dog,gausfilt);
im42cor = harcor(im42g,dog,gausfilt);

%imshow([im11g im12g]);
%title('im11g  im12g');figure;
%imshow([im21g im22g]);
%title('im21g  im22g');figure;

fprintf('FAST:\n');

%image 1 FAST:
tic
im11o = my_fast_detector(im11g,ones(size(im11g)));
toc
imwrite(im11o,'S1-fast.png')
tic
im12o = my_fast_detector(im12g,ones(size(im12g)));
toc
imshow([im11o+im11g im12o+im12g]);
title('im11o im12o');figure;

%image 2 FAST:
tic
im21o = my_fast_detector(im21g,ones(size(im21g)));
toc
imwrite(im21o,'S2-fast.png')
tic
im22o = my_fast_detector(im22g,ones(size(im22g)));
toc
%imshow([im21o+im21g im22o+im22g]);
%title('im21o im22o');figure;
tic
im23o = my_fast_detector(im23g,ones(size(im23g)));
toc
tic
im24o = my_fast_detector(im24g,ones(size(im24g)));
toc
imshow([im21o+im21g im22o+im22g im23o+im23g im24o+im24g]);
title('im21o  im22o  im23o  im24o');figure;

%image 3 fast:
tic
im31o = my_fast_detector(im31g,ones(size(im31g)));
toc
imwrite(im31o,'S3-fast.png')
tic
im32o = my_fast_detector(im32g,ones(size(im32g)));
toc
%imshow([im31o+im31g im32o+im32g]);
%title('im31o im32o');figure;
tic
im33o = my_fast_detector(im33g,ones(size(im33g)));
toc
tic
im34o = my_fast_detector(im34g,ones(size(im34g)));
toc
imshow([im31o+im31g im32o+im32g im33o+im33g im34o+im34g]);
title('im31o  im32o  im33o  im34o');figure;

%image 4 fast:
tic
im41o = my_fast_detector(im41g,ones(size(im41g)));
toc
imwrite(im41o,'S4-fast.png')
tic
im42o = my_fast_detector(im42g,ones(size(im42g)));
toc
imshow([im41o+im41g im42o+im42g]);
title('im41o im42o');figure;



%
fprintf('FASTR:\n')

%fastr:
fastrThresh = 0.0005;

%image 1 fastr:
tic
%im11or = im11o .* im11cor;
im11or = my_fast_detector(im11g,im11cor);
im11or= im11or.*im11cor>fastrThresh;
toc
%imshow(im11or)
%title('im11or');figure;
imwrite(im11or,'S1-fastR.png');
tic
%im12or = im12o .* im12cor;
im12or = my_fast_detector(im12g,im12cor);
im12or= im12or.*im12cor>fastrThresh;
%imshow(im12or)
%title('im12or');figure;
toc
imshow([im11or+im11g im12or+im12g]);
title('im11or im12or');figure;


%image 2 fastr:
tic
%im21or = im21o .* im21cor;
im21or = my_fast_detector(im21g,im21cor);
im21or= im21or.*im21cor>fastrThresh;
%imshow(im21or)
%title('im21or');figure;
toc
imwrite(im21or,'S2-fastR.png');
tic
im22or = my_fast_detector(im22g,im22cor);
im22or = im22or .* im22cor;
im22or= im22or>fastrThresh;
%imshow(im22or)
%title('im22or');figure;
%imshow([im21or+im21g im22or+im22g]);
%title('im21or im22or');figure;
toc
tic
im23or = my_fast_detector(im23g,im23cor);
im23or = im23or .* im23cor;
im23or= im23or>fastrThresh;
toc
%imshow(im23or)
%title('im23or');figure;
tic
im24or = my_fast_detector(im24g,im24cor);
im24or = im24or .* im24cor;
im24or= im24or>fastrThresh;
toc
%imshow(im24or)
%title('im24or');figure;
imshow([im21or+im21g im22or+im22g im23or+im23g im24or+im24g]);
title('im21or  im22or  im23or  im24or');figure;


%image 3 fastr: 
tic
im31or = my_fast_detector(im31g,im31cor);
im31or = im31or .* im31cor;
im31or= im31or>fastrThresh;
toc
%imshow(im31or)
%title('im31or');figure;
imwrite(im31or,'S3-fastR.png');
tic
im32or = my_fast_detector(im32g,im32cor);
im32or = im32or .* im32cor;
im32or= im32or>fastrThresh;
toc
%imshow(im32or)
%title('im32or');figure;
%imshow([im31or+im31g im32or+im32g]);
%title('im31or im32or');figure;
tic
im33or = my_fast_detector(im33g,im33cor);
im33or = im33or .* im33cor;
im33or= im33or>fastrThresh;
toc
%imshow(im33or)
%title('im33or');figure;
tic
im34or = my_fast_detector(im34g,im34cor);
im34or = im34or .* im34cor;
im34or= im34or>fastrThresh;
toc
%imshow(im34or)
%title('im34or');figure;
imshow([im31or+im31g im32or+im32g im33or+im33g im34or+im34g]);
title('im31or  im32or  im33or  im34or');figure;


%image 4 fastr:
tic
%im41or = im41o .* im41cor;
im41or = my_fast_detector(im41g,im41cor);
im41or= im41or.*im41cor>fastrThresh;
toc
%imshow(im41or)
%title('im41or');figure;
imwrite(im41or,'S4-fastR.png');
tic
%im42or = im42o .* im42cor;
im42or = my_fast_detector(im42g,im42cor);
im142or= im42or.*im42cor>fastrThresh;
%imshow(im42or)
%title('im42or');figure;
toc
imshow([im41or+im41g im42or+im42g]);
title('im41or im42or');figure;


% FASTR method works about 1second (about 30% quicker) than the FAST method


% im1(:,:,:,1) = im11; to access individual color photo etc
% 
%RANSAC parameter
confidence = 97;
trials = 1000;

% image set 1
points11 = detectSURFFeatures(im11g);
[features11, points11] = extractFeatures(im11g,points11);
max1 = 2;
tforms(max1) = projtform2d;
imageSize=zeros(max1,2);
imageSize(2,:) = size(im12g);
points12 = detectSURFFeatures(im12g);
[features12, points12] = extractFeatures(im12g,points12);
iPairs1 = matchFeatures(features12,features11);
%matches
ix=max(size(iPairs1));
matches = zeros(size(im11g));
for i=1:ix
    if iPairs1(i,1)<351
        matches(iPairs1(i,1),iPairs1(i,2))=1;
    end
end
%imshow([im12g+matches]);

matchedPts = points12(iPairs1(:,1),:);
matchedPtsPrev = points11(iPairs1(:,2),:);
matchedsurf = showMatchedFeatures(im11g,im12g,matchedPtsPrev,matchedPts,'montage');
%imshow(matchedsurf);
title('matches');
%imwrite(matchedsurf,'S1-fastmatch.png');
figure;
tforms(2) = estgeotform2d(matchedPts,matchedPtsPrev,'projective','Confidence',confidence,'MaxNumTrials',trials);
tforms(2).A = tforms(1).A * tforms(2).A; 
[xlim(1,:), ylim(1,:)] = outputLimits(tforms(1), [1 imageSize(1,2)], [1 imageSize(1,1)]);
[xlim(2,:), ylim(2,:)] = outputLimits(tforms(2), [1 imageSize(2,2)], [1 imageSize(2,1)]);

avgXLim = mean(xlim, 2);
[lowlow,idx] = sort(avgXLim);
baseImNum = floor((numel(max1)+1)/2);
baseIm = idx(baseImNum);
Tinv = invert(tforms(baseIm));
tforms(1).A = Tinv.A * tforms(1).A;
tforms(2).A = Tinv.A * tforms(2).A;
[xlim(1,:), ylim(1,:)] = outputLimits(tforms(1), [1 imageSize(1,2)], [1 imageSize(1,1)]);
[xlim(2,:), ylim(2,:)] = outputLimits(tforms(2), [1 imageSize(2,2)], [1 imageSize(2,1)]);
maxImageSize = max(imageSize);
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);
width  = round(xMax-xMin);
height = round(yMax-yMin);
pan1 = zeros([height width 3], 'like', im12);
blender = vision.AlphaBlender('Operation', 'Binary mask','MaskSource', 'Input port');
xLimits = [xMin xMax];
yLimits = [yMin yMax];
pan1View = imref2d([height width], xLimits, yLimits);
tformdImage = imwarp(im11, tforms(1), 'OutputView', pan1View);
mask = imwarp(true(size(im11,1),size(im11,2)), tforms(1), 'OutputView', pan1View);
pan1 = step(blender, pan1, tformdImage, mask);
tformdImage2 = imwarp(im12, tforms(2), 'OutputView', pan1View);
mask2 = imwarp(true(size(im12,1),size(im12,2)), tforms(2), 'OutputView', pan1View);
pan1 = step(blender, pan1, tformdImage2, mask2);
imshow(pan1);
title('pano 1');figure;
imwrite(pan1,'S1-panorama.png')


% image set 2v2:
clear im1
max2=4;%max(size(im2(1,1,1,:)));
points2 = detectSURFFeatures(im2g(:,:,1));
features2 = zeros(1,max2);
[features2, points2] = extractFeatures(im2g(:,:,1),points2);
tforms(max2) = projtform2d;
imageSize=zeros(max2,2);
temp =1;
%loop
for num = 2:max2
    imageSize(num,:) = size(im2g(:,:,num));
    ppoints2 = points2;
    pfeatures2 = features2;
    points2 = detectSURFFeatures(im2g(:,:,num));
    [features2, points2] = extractFeatures(im2g(:,:,num),points2);
    iPairs2 = matchFeatures(features2,pfeatures2);
    matchedPts = points2(iPairs2(:,1),:);
    matchedPtsPrev = ppoints2(iPairs2(:,2),:);

    if temp==1
        matchedsurf2 = showMatchedFeatures(im21g,im22g,matchedPtsPrev,matchedPts,'montage');
        %imshow(matchedsurf2);
        title('matches2');
        %imwrite(matchedsurf2,'S2-fastmatch.png');
        figure;

        temp=0;
    end

    tforms(num) = estgeotform2d(matchedPts,matchedPtsPrev,'projective','Confidence',confidence,'MaxNumTrials',trials);
    tforms(num).A = tforms(num-1).A * tforms(num).A; 
end
for num = 1:max2
    [xlim(num,:), ylim(num,:)] = outputLimits(tforms(num), [1 imageSize(num,2)], [1 imageSize(num,1)]);
end
avgXLim = mean(xlim, 2);
[lowlow,idx] = sort(avgXLim);
baseImNum = floor((numel(tforms)+1)/2);
baseIm = idx(baseImNum);
Tinv = invert(tforms(baseIm));
for num = 1:max2   
    tforms(num).A = Tinv.A * tforms(num).A;
    [xlim(num,:), ylim(num,:)] = outputLimits(tforms(num), [1 imageSize(num,2)], [1 imageSize(num,1)]);
end
maxImageSize = max(imageSize);
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);
width  = round(xMax-xMin);
height = round(yMax-yMin);
pan2 = zeros([height width 3], 'like', im2(:,:,:,1));
blender = vision.AlphaBlender('Operation','Binary mask','MaskSource', 'Input port');  
xLimits = [xMin xMax];
yLimits = [yMin yMax];
pan2View = imref2d([height width], xLimits, yLimits);
for num = 1:max2
    tformdImage = imwarp(im2(:,:,:,num), tforms(num), 'OutputView', pan2View);
    mask = imwarp(true(size(im2(:,:,:,num),1),size(im2(:,:,:,num),2)), tforms(num), 'OutputView', pan2View);
    pan2 = step(blender, pan2, tformdImage, mask);
end
imshow(pan2);
title('pano2');figure;
imwrite(pan2,'S2-panorama.png')



clear im2
% image set 3:
max3=4;%max(size(im3(1,1,1,:)));
points3 = detectSURFFeatures(im3g(:,:,1));
features3 = zeros(1,max3);
[features3, points3] = extractFeatures(im3g(:,:,1),points3);
tforms(max3) = projtform2d;
imageSize=zeros(max3,2);
%loop
for num = 2:max3
    imageSize(num,:) = size(im3g(:,:,num));
    ppoints3 = points3;
    pfeatures3 = features3;
    points3 = detectSURFFeatures(im3g(:,:,num));
    [features3, points3] = extractFeatures(im3g(:,:,num),points3);
    iPairs3 = matchFeatures(features3,pfeatures3);
    matchedPts = points3(iPairs3(:,1),:);
    matchedPtsPrev = ppoints3(iPairs3(:,2),:);
    tforms(num) = estgeotform2d(matchedPts,matchedPtsPrev,'projective','Confidence',confidence,'MaxNumTrials',trials);
    tforms(num).A = tforms(num-1).A * tforms(num).A; 
end
for num = 1:max3
    [xlim(num,:), ylim(num,:)] = outputLimits(tforms(num), [1 imageSize(num,2)], [1 imageSize(num,1)]);
end
avgXLim = mean(xlim, 2);
[lowlow,idx] = sort(avgXLim);
baseImNum = floor((numel(tforms)+1)/2);
baseIm = idx(baseImNum);
Tinv = invert(tforms(baseIm));
for num = 1:max3   
    tforms(num).A = Tinv.A * tforms(num).A;
    [xlim(num,:), ylim(num,:)] = outputLimits(tforms(num), [1 imageSize(num,2)], [1 imageSize(num,1)]);
end
maxImageSize = max(imageSize);
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);
width  = round(xMax-xMin);
height = round(yMax-yMin);
pan3 = zeros([height width 3], 'like', im3(:,:,:,1));
blender = vision.AlphaBlender('Operation','Binary mask','MaskSource', 'Input port');  
xLimits = [xMin xMax];
yLimits = [yMin yMax];
pan3View = imref2d([height width], xLimits, yLimits);
for num = 1:max3
    tformdImage = imwarp(im3(:,:,:,num), tforms(num), 'OutputView', pan3View);
    mask = imwarp(true(size(im3(:,:,:,num),1),size(im3(:,:,:,num),2)), tforms(num), 'OutputView', pan3View);
    pan3 = step(blender, pan3, tformdImage, mask);
end
imshow(pan3);
title('pano3');figure;
imwrite(pan3,'S3-panorama.png')
%

%
% image set 4
points41 = detectSURFFeatures(im41g);
[features41, points41] = extractFeatures(im41g,points41);
max4 = 2;
tforms(max4) = projtform2d;
imageSize=zeros(max4,2);
imageSize(2,:) = size(im42g);
points42 = detectSURFFeatures(im42g);
[features42, points42] = extractFeatures(im42g,points42);
iPairs4 = matchFeatures(features42,features41);
%matches
ix=max(size(iPairs4));
matches = zeros(size(im41g));
for i=1:ix
    if iPairs4(i,1)<351
        matches(iPairs4(i,1),iPairs4(i,2))=1;
    end
end
%imshow([im12g+matches]);

matchedPts = points42(iPairs4(:,1),:);
matchedPtsPrev = points41(iPairs4(:,2),:);
matchedsurf = showMatchedFeatures(im41g,im42g,matchedPtsPrev,matchedPts,'montage');
%imshow(matchedsurf);
title('matches');
%imwrite(matchedsurf,'S1-fastmatch.png');
figure;
tforms(2) = estgeotform2d(matchedPts,matchedPtsPrev,'projective','Confidence',confidence,'MaxNumTrials',trials);
tforms(2).A = tforms(1).A * tforms(2).A; 
[xlim(1,:), ylim(1,:)] = outputLimits(tforms(1), [1 imageSize(1,2)], [1 imageSize(1,1)]);
[xlim(2,:), ylim(2,:)] = outputLimits(tforms(2), [1 imageSize(2,2)], [1 imageSize(2,1)]);

avgXLim = mean(xlim, 2);
[lowlow,idx] = sort(avgXLim);
baseImNum = floor((numel(max4)+1)/2);
baseIm = idx(baseImNum);
Tinv = invert(tforms(baseIm));
tforms(1).A = Tinv.A * tforms(1).A;
tforms(2).A = Tinv.A * tforms(2).A;
[xlim(1,:), ylim(1,:)] = outputLimits(tforms(1), [1 imageSize(1,2)], [1 imageSize(1,1)]);
[xlim(2,:), ylim(2,:)] = outputLimits(tforms(2), [1 imageSize(2,2)], [1 imageSize(2,1)]);
maxImageSize = max(imageSize);
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);
width  = round(xMax-xMin);
height = round(yMax-yMin);
pan4 = zeros([height width 3], 'like', im42);
blender = vision.AlphaBlender('Operation', 'Binary mask','MaskSource', 'Input port');
xLimits = [xMin xMax];
yLimits = [yMin yMax];
pan4View = imref2d([height width], xLimits, yLimits);
tformdImage = imwarp(im41, tforms(1), 'OutputView', pan4View);
mask = imwarp(true(size(im41,1),size(im41,2)), tforms(1), 'OutputView', pan4View);
pan4 = step(blender, pan4, tformdImage, mask);
tformdImage2 = imwarp(im42, tforms(2), 'OutputView', pan4View);
mask2 = imwarp(true(size(im42,1),size(im42,2)), tforms(2), 'OutputView', pan4View);
pan4 = step(blender, pan4, tformdImage2, mask2);
imshow(pan4);
title('pano 4');figure;
imwrite(pan4,'S4-panorama.png')







%functions --------- \/


function out = my_fast_detector(im,rmat)
    
    checkspots = [0 0 1 1 1 0 0; 0 1 0 0 0 1 0; 1 0 0 0 0 0 1;1 0 0 1 0 0 1;1 0 0 0 0 0 1;0 1 0 0 0 1 0;0 0 1 1 1 0 0];
    % corners1 5 9 13 = (1,4),(4,7),(7,4),(4,1)
    % nnz();
    ip = [0 0];
    [maxy maxx] = size(im);
    n=9;
    t=0.08;
    testfeatmat = zeros(size(im));
    testfeatmat2 = zeros(size(im));
    testfeatmat3 = zeros(size(im));
    featmat = zeros(7,7);
    orderedcirc = ones(1,16);
    cornermat = zeros(size(im));
    
    for pixy = 4:maxy-3
        for pixx = 4:maxx-3
            if rmat(pixy,pixx)>0
                ip = [pixy pixx];
                featmat = im(pixy-3:pixy+3,pixx-3:pixx+3).*checkspots;
                featmat2 = im(pixy-3:pixy+3,pixx-3:pixx+3).*checkspots > im(pixy,pixx)+t;
                %featmat3 = im(pixy-3:pixy+3,pixx-3:pixx+3).*checkspots < im(pixy,pixx)-t
                orderedcirc(1)=featmat(4,7);
                orderedcirc(2)=featmat(5,7);
                orderedcirc(3)=featmat(6,6);
                orderedcirc(4)=featmat(7,5);
                orderedcirc(5)=featmat(7,4);
                orderedcirc(6)=featmat(7,3);
                orderedcirc(7)=featmat(6,2);
                orderedcirc(8)=featmat(5,1);
                orderedcirc(9)=featmat(4,1);
                orderedcirc(10)=featmat(3,1);
                orderedcirc(11)=featmat(2,2);
                orderedcirc(12)=featmat(1,3);
                orderedcirc(13)=featmat(1,4);
                orderedcirc(14)=featmat(1,5);
                orderedcirc(15)=featmat(2,6);
                orderedcirc(16)=featmat(3,7);
                testfeatmat(pixy,pixx) = nnz(orderedcirc>im(pixy,pixx)+t);
                testfeatmat2(pixy,pixx) = nnz(orderedcirc<im(pixy,pixx)-t);
                
                
                
                for seqq = 1:16
                    orderedcirctest=orderedcirc>(im(pixy,pixx)+t);
                    orderedcirctest2=orderedcirc<(im(pixy,pixx)-t);
                    if nnz(orderedcirctest(seqq:mod((seqq+n),16)))==n
                        cornermat(pixy,pixx)=1;
                    elseif nnz(orderedcirctest2(seqq:mod((seqq+n),16)))==n
                        cornermat(pixy,pixx)=1;
                    end
                    %starto=1;
                    %startom = 1;
                    %for seqn = 1:n
                        
                    %    starto = starto * orderedcirc(mod(seqq+seqn,16)+1)>(im(pixy,pixx)+t);
                    %    startom = startom * orderedcirc(mod(seqq+seqn,16)+1)<(im(pixy,pixx)-t);
                        %check n orderedcirc items are same
                        %(orderedcirc>im(pixy,pixx)+t) etc
                     %   if starto==0
                     %       if startom==0
                    %            continue
                    %        end
                     %   end
                    %end
    
                end
            end
        end
    end
    %imshow(testfeatmat2)
    %title('idfk2');figure;
    %imshow(testfeatmat);
    %title('idfk');figure;
    poscorn = testfeatmat>=n;
    %out = poscorn;
    %imshow(cornermat);
    %title('cornermat');figure;
    out = cornermat;
end

function icorn = harcor(im,dog,gausfilt)
    imx = imfilter(im,dog);
    imy = imfilter(im,dog');
    imx2g = imfilter(imx .* imx, gausfilt);
    imy2g = imfilter(imy .* imy, gausfilt);
    imximyg = imfilter(imx .* imy, gausfilt);
    icorn = imx2g.*imy2g - imximyg .* imximyg - 0.05*(imx2g+imy2g).^2;
    %imshow(icorn*50);
    %title('icorn');figure;
end





