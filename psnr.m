close all;
clear all;
clc;
fileFolder=fullfile('/home/yyl/pjs/pycharm-remote/MyGAN/test_out_imgs/20180317-170513');
dirOutput=dir(fullfile(fileFolder,'*.png'));
LengthFiles = length(dirOutput);
RESULT_sr=zeros(1,LengthFiles);
RESULT_bic=zeros(1,LengthFiles);
for i=0:LengthFiles-1
    filename = dirOutput(i+1).name;
    img=imread(filename);
    [h w c]=size(img);
    w = w/3;
    img_bicubic=img(1:h,1:w,1:c);
    img_sr=img(:,129:256,:);
    img_hr=img(:,257:384,:);

    B=8;
    MAX=2^B-1;       
    MES_sr=sum(sum((img_sr-img_hr).^2))/(h*w);     
    MES_bic=sum(sum((img_bicubic-img_hr).^2))/(h*w);
    PSNR_sr =20*log10(MAX/sqrt(MES_sr));
    PSNR_bic = 20*log10(MAX/sqrt(MES_bic));
    RESULT_sr(i+1) = (PSNR_sr(:,:,1) + PSNR_sr(:,:,2) + PSNR_sr(:,:,3))/3;
    RESULT_bic(i+1) = (PSNR_bic(:,:,1) + PSNR_bic(:,:,2) + PSNR_bic(:,:,3))/3;
end

plot(1:LengthFiles,RESULT_sr,1:LengthFiles,RESULT_bic);
mean_bic = mean(RESULT_bic)
mean_sr = mean(RESULT_sr)

