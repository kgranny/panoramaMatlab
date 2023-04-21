%% cmpt 361
%assignment 1
% By: Kyle Granville

close all
clear all
clc

darker = 30;
imsize = 500;

hp = imread('bricks.png');
lp = imread('plane.png');

hp = im2double(hp);
lp = im2double(lp);

hp = rgb2gray(hp);
lp = rgb2gray(lp);
hp = imresize(hp,[imsize imsize]);
lp = imresize(lp,[imsize imsize]);
imwrite(hp,'HP.png');
imwrite(lp,'LP.png');

sobelkern = [-1 0 1; -2 0 2; -1 0 1];

gauskern = fspecial('gaussian',21,2.5);
surf(gauskern)
title('gaussian surf');
saveas(gcf,'gaus-surf.png');
figure;

dog = conv2(gauskern,sobelkern);
surf(dog);
title('dog surf');
saveas(gcf,'dog-surf.png');
figure;

% gaus HP
hpgaus = imfilter(hp,gauskern);
imwrite(hpgaus,'HP-filt.png');

hpfreq = fft2(hp);
hpfreq = abs(fftshift(hpfreq))/darker;
imwrite(hpfreq,'HP-freq.png');

hpfreqfilt = fft2(hpgaus);
hpfreqfilt = abs(fftshift(hpfreqfilt))/darker;
imshow(hpfreqfilt);
title('HP filt freq');
imwrite(hpfreqfilt,'HP-filt-freq.png');
figure;

%gaus LP

lpgaus = imfilter(lp,gauskern);
imwrite(hpgaus,'LP-filt.png');

lpfreq = fft2(lp);
lpfreq = abs(fftshift(lpfreq))/darker;
imwrite(lpfreq,'LP-freq.png');

lpfreqfilt = fft2(lpgaus);
lpfreqfilt = abs(fftshift(lpfreqfilt))/darker;
imshow(lpfreqfilt);
title('LP filt freq');
imwrite(lpfreqfilt,'LP-filt-freq.png');
figure;

%dog LP
hpdog = imfilter(hp,dog);
imshow(hpdog);
title('hp dog');
imwrite(hpdog,'HP-dogfilt.png');
figure;

hpdogfreq = fft2(hpdog);
hpdogfreq = abs(fftshift(hpdogfreq))/darker;
imshow(hpdogfreq);
title('hp dog freq');
imwrite(hpdogfreq,'HP-dogfilt-freq.png');
figure;

lpdog = imfilter(lp,dog);
imshow(lpdog);
title('lp dog');
imwrite(lpdog,'LP-dogfilt.png');
figure;

lpdogfreq = fft2(lpdog);
lpdogfreq = abs(fftshift(lpdogfreq))/darker;
imshow(lpdogfreq);
title('lp dog freq');
imwrite(lpdogfreq,'LP-dogfilt-freq.png');

% alias/subsampling
% ss2
g2h = fspecial('gaussian',11,1.2);
g2l = fspecial('gaussian',11,1.2);
%surf(g2)
%figure
hpg2 = imfilter(hp,g2h);
lpg2 = imfilter(lp,g2l);

hpss2 = hp(1:2:end,1:2:end);
imwrite(hpss2,'HP-sub2.png');
lpss2 = lp(1:2:end, 1:2:end);
imwrite(lpss2,'LP-sub2.png');

hpfreqss2 = hpfreq(1:2:end,1:2:end);
imwrite(hpfreqss2,'HP-sub2-freq.png');
lpfreqss2 = lpfreq(1:2:end,1:2:end);
imwrite(lpfreqss2,'LP-sub2-freq.png');

hpss2_filt = hpg2(1:2:end, 1:2:end);
imwrite(hpss2_filt,'HP-sub2-aa.png');
lpss2_filt = lpg2(1:2:end, 1:2:end);
imwrite(lpss2_filt,'LP-sub2-aa.png');

hp_sub2_aa_freq = abs(fftshift(fft2(hpss2_filt)))/darker;
imshow(hp_sub2_aa_freq);
imwrite(hp_sub2_aa_freq,'HP-sub2-aa-freq.png');
title('lp sub2 aa freq')
figure

lp_sub2_aa_freq = abs(fftshift(fft2(lpss2_filt)))/darker;
imshow(lp_sub2_aa_freq);
imwrite(lp_sub2_aa_freq,'LP-sub2-aa-freq.png');
title('lp sub2 aa freq')
figure


imshow([imresize(hpss2,[500 500]) imresize(hpss2_filt,[500 500])]);
title('hp subsampled2 vs subsampled2_filt');
figure
imshow([imresize(lpss2,[500 500]) imresize(lpss2_filt,[500 500])]);
title('lp subsampled2 vs subsampled2_filt');
figure
%ss4
g4 = fspecial('gaussian',15,1.5);
surf(g4)
figure
hpg4 = imfilter(hp,g4);
lpg4 = imfilter(lp,g4);

hpss4 = hp(1:4:end,1:4:end);
imwrite(hpss4,'HP-sub4.png');
lpss4 = lp(1:4:end, 1:4:end);
imwrite(lpss4,'LP-sub4.png');

hpfreqss4 = hpfreq(1:4:end,1:4:end);
imwrite(hpfreqss4,'HP-sub4-freq.png');
lpfreqss4 = lpfreq(1:4:end,1:4:end);
imwrite(lpfreqss4,'LP-sub4-freq.png');

hpss4_filt = hpg4(1:4:end, 1:4:end);
imwrite(hpss4_filt,'HP-sub4-aa.png');
lpss4_filt = lpg4(1:4:end, 1:4:end);
imwrite(lpss4_filt,'LP-sub4-aa.png');

hp_sub4_aa_freq = abs(fftshift(fft2(hpss4_filt)))/darker;
imshow(hp_sub4_aa_freq);
imwrite(hp_sub4_aa_freq,'HP-sub4-aa-freq.png');
title('lp sub4 aa freq')
figure

lp_sub4_aa_freq = abs(fftshift(fft2(lpss4_filt)))/darker;
imshow(lp_sub4_aa_freq);
imwrite(lp_sub4_aa_freq,'LP-sub4-aa-freq.png');
title('lp sub4 aa freq')
figure

imshow([imresize(hpss4,[500 500]) imresize(hpss4_filt,[500 500])]);
figure
imshow([imresize(lpss4,[500 500]) imresize(lpss4_filt,[500 500])]);
figure


% canny 
canopt = edge(hpgaus,'canny',0.16);
subplot(2,3,1)
imshow(canopt)
imwrite(canopt,'HP-canny-optimal.png');
    
canlowlow = edge(hpgaus,'canny',0.03);
subplot(2,3,2)
imshow(canlowlow)
imwrite(canlowlow,'HP-canny-lowlow.png');

canhighlow = edge(hpgaus,'canny',0.05);
subplot(2,3,3)
imshow(canhighlow)
imwrite(canhighlow,'HP-canny-highlow.png');
    
canlowhigh = edge(hpgaus,'canny',0.2);
subplot(2,3,4)
imshow(canlowhigh)
imwrite(canlowhigh,'HP-canny-lowhigh.png');

canhighhigh = edge(hpgaus,'canny',0.25);
subplot(2,3,5)
imshow(canhighhigh)
imwrite(canhighhigh,'HP-canny-highhigh.png');
figure 
%
canoptl = edge(lpgaus,'canny',0.26);
subplot(2,3,1)
imshow(canoptl)
imwrite(canoptl,'LP-canny-optimal.png');
    
canlowlowl = edge(lpgaus,'canny',0.2);
subplot(2,3,2)
imshow(canlowlowl)
imwrite(canlowlowl,'LP-canny-lowlow.png');

canhighlowl = edge(lpgaus,'canny',0.25);
subplot(2,3,3)
imshow(canhighlowl)
imwrite(canhighlowl,'LP-canny-highlow.png');

canlowhighl = edge(lpgaus,'canny',0.27);
subplot(2,3,4)
imshow(canlowhighl)
imwrite(canlowhighl,'LP-canny-lowhigh.png');

canhighhighl = edge(lpgaus,'canny',0.35);
subplot(2,3,5)
imshow(canhighhighl)
imwrite(canhighhighl,'LP-canny-highhigh.png');



