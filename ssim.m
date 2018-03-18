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
    img_bic=img(1:h,1:w,1:c);
    img_sr=img(:,129:256,:);
    img_hr=img(:,257:384,:);
    
    RESULT_sr(i+1) = SSIM(img_sr,img_hr)
    RESULT_bic(i+1) = SSIM(img_bic,img_hr)
    
end

plot(1:LengthFiles,RESULT_sr,1:LengthFiles,RESULT_bic);
mean_bic = mean(RESULT_bic)
mean_sr = mean(RESULT_sr)


function re=SSIM(X,Y)
    X=double(X);
    Y=double(Y);

    ux=mean(mean(mean(X)));
    uy=mean(mean(mean(Y)));

    sigma2x=mean(mean(mean((X-ux).^2)));
    sigma2y=mean(mean(mean((Y-uy).^2)));   
    sigmaxy=mean(mean(mean((X-ux).*(Y-uy))));

    k1=0.01;
    k2=0.03;
    L=255;
    c1=(k1*L)^2;
    c2=(k2*L)^2;
    c3=c2/2;

    l=(2*ux*uy+c1)/(ux*ux+uy*uy+c1);
    c=(2*sqrt(sigma2x)*sqrt(sigma2y)+c2)/(sigma2x+sigma2y+c2);
    s=(sigmaxy+c3)/(sqrt(sigma2x)*sqrt(sigma2y)+c3);

    re=l*c*s;

end

