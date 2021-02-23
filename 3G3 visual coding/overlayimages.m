function overlayimages(image,B,dimage)


A=imfilt(image,B);

a_max=max(max(image));
a_min=min(min(image));
clf
colormap(gray);
image_size=sqrt(size(image,1));

clf
A=(A-min(A))/range(A);

m1=min(image);
m2=max(image);

this_image=(reshape(image,image_size,image_size))';
this_image(end,end,3)=0;
this_image(:,:,2)=this_image(:,:,1);
this_image(:,:,3)=this_image(:,:,1);

overlay_image=1-reshape(A,image_size,image_size)';
bw=1-reshape(A,image_size,image_size)';
bw(:,:,2)= bw(:,:,1);
bw(:,:,3)= bw(:,:,1);

overlay_image(end,end,3)=0;
if nargin==3
this_image=(reshape(dimage,image_size,image_size))';
this_image(end,end,3)=0;
this_image(:,:,2)=this_image(:,:,1);
this_image(:,:,3)=this_image(:,:,1);
end
subplot(2,2,1)
imagesc(this_image)
title('original image');

subplot(2,2,2)
imagesc(bw)
title('filtered image');

subplot(2,2,3)
imagesc(this_image.*overlay_image)
title('filter overlayed on original');

subplot(2,2,4)
Bs=reshape(B,sqrt(length(B)),sqrt(length(B)))';

imagesc(Bs)
title('filter (base)');

axis off
axis square






