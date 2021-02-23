%pca:  perfoemc principal conones on images
%[coeff, percent]=pca(x, sz,nsamp)
%retrurns the prinicpla components  coeff obtained by  perofming 
%principal compinents on nsamp samples of sz x sz patches of
%images obtained from the images in  x (columns are vectorized images)
% percent is the percentage explained by each principal component

function [coeff, percent]=pca(images, sz,nsamp)

image_size=sqrt(size(images,1));
nimages=size(images,2);
pix=sz*sz;
p=zeros(pix,nsamp*nimages);


w=0;
for k=1:nimages
    this_image=reshape(images(:,k),image_size,image_size)';
    this_image=rot90(this_image,-1.0);

    for j=1:nsamp
        w=w+1;
        pp=ceil(rand*(image_size-sz));
        qq=ceil(rand*(image_size-sz));
        d=this_image(pp:pp+sz-1,qq:qq+sz-1);      
        p(:,w)=reshape(double(d),numel(d),1);
    end
    k;
end


% Perform a singular value decomposition on the image samples
%[u, s, v] = svd(p, 0);
 [coeff, score,latent] = pca(p');
 percent=100*latent/sum(latent);



