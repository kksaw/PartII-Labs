function S=extract_subimages(images,nsub,sz)  
image_size=sqrt(size(images,1));
num_images=size(images,2);
npix=sz^2;
i=ceil(num_images*rand);    % choose an image for this step
    this_image=reshape(images(:,i),image_size,image_size)'; % turn it into a matrix

    % extract subimages at random from this image to make swatch matrix S
    for i=1:nsub
        r=ceil((image_size-sz)*rand);
        c=ceil((image_size-sz)*rand);
        S(:,i)=reshape(this_image(r:r+sz-1,c:c+sz-1),npix,1);
    end