function B= sparseopt(images,nsteps)
clear functions % this is needed to clear variables that are within the  subroutines

num_images=size(images,2); %this is the number of images, that is columns
image_size=sqrt(size(images,1)); %image is a square array image_size x image_size

nbase=64; % number of bases
sz=8; %linear size of each base
npix=sz*sz; % number of pixels in base
nsubimages=100;% number of swatches to extract each cycle

%initialise bases to small random numbers
B = rand(npix,nbase)-0.5;

sigma=0.316; %from sparseness cost in equation 1.2
lambda=100; % from trade-off of sparseness and reconstruction cost in equation 1.2
eta = 1.0; %step size for base leanring rule in equation 1.3

%initlaise a zero matrix to put subimages into
S=zeros(npix,nsubimages);

display_every=50; % how often to update display

%%%% Main  optimization loop 

for t=1:nsteps
    %extract  nsubimages subimages of linear dimension sz randonly from the images
    S=extract_subimages(images,nsubimages,sz);
    
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%Inner loop start %%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   
    % calculate activations for this swatch matrix via conjugate gradient routine
    A= fast_minimize(B,S,sigma,lambda);

    % update bases
    dB=zeros(npix,nbase); %zero the update

    %sum the changes over the subimages
    for i=1:nsubimages
        dB = dB + basefunc(A(:,i),B,S(:,i));
    end
    %update bases
    B = B - eta*dB/nsubimages;

     
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%Inner loop end %%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %normalization
    B=normalize_bases(B,A);
   
    %display progress as the bases
    if (mod(t,display_every)==1)
        fprintf('Cycle number %i\n',t)
        displayimages(B,64);
        pause(0.1)
        drawnow;
    end
end

%normalise the bases
B = B*diag(1./sqrt(sum(B.*B)));
