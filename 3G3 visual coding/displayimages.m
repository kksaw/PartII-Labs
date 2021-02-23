%% Display the images
%displayimages: displays images
%displayimages(x,num)
%plots the first num images 
% each column  of x contains a vectorized image which must contain a
% square number of elements  representing a square image

function displayimages(images,num)

if num > size(images,2)
    error('Can not display %2.0f images as there are only %2.0f ',num,size(images,2));
end

if sqrt(num)==floor(sqrt(num))
    n=sqrt(num);
    m=n;
else
    n=ceil(sqrt(num));
    if n*(n-1)>=num
        m=n-1;
    else m=n;
    end
end


a_max=max(max(images));
a_min=min(min(images));
clf
colormap(gray);
image_size=sqrt(size(images,1));
q=min(size(images,2),n*m);
for k=1:num
    subplot(m,n,k)
    this_image=reshape(images(:,k),image_size,image_size)';
    imagesc(this_image);
    %,[-a_max a_max]);
    axis off   
    axis square
end





