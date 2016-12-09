function [ ] = imageTiling( image_fullfname, tile_size, factor, ...
    masks_fullfnames, tiles_path)
%% imageTiling  cropping an image to tiles and saving them to files
%   The fucntion crops a given image and several class masks to image tiles
%   which are saved as image files in folders corresponding to the masks
%   image_fullname - the full name of the image to be croped/tiles
%   tile_size - vector for 2 elements- the number of rows and columns of the tile
%   factor - the factor for the minimum number of pixels from all pixels with given label
%   masks_fullfnames -cel array of the full names of the binary masks corresponding to
%                     the class label 1:'Slum', 2: 'Urban' and 3: 'Rural'
%   tiles_path - the main filder for the tiles to be saved in subfolders
% For Testing use test_imageTiling

%% input control
if nargin < 5
    error('imageTiling: not enough input arguments!');
end

%% params -> vars
slum_mask_fname = char(masks_fullfnames{1});
urban_mask_fname = char(masks_fullfnames{2});
rural_mask_fname = char(masks_fullfnames{3});

% basename and extention for the tiles
[~,base_fname,ext] = fileparts(image_fullfname);  
ext = ext(2:end);

%% load the data from the files
image_data = imread(image_fullfname);
slum_mask = imread(slum_mask_fname);
urban_mask = imread(urban_mask_fname);
rural_mask = imread(rural_mask_fname);

%% dimensions
nrows = size(image_data, 1); ncols = size(image_data, 2); 
nrows_tile = tile_size(1); ncols_tile = tile_size(2);

%% tiling
for sr = 1: nrows_tile : nrows
    er = min(nrows, sr + nrows_tile - 1);
    for sc = 1: ncols_tile : ncols
        ec = min(ncols, sc + ncols_tile - 1);
        extent = [sr er sc ec];
        
        % get the image and masks tile
        image_tile = image_data(sr:er, sc:ec, :);
        slum_tile  = slum_mask(sr:er, sc:ec,:);
        urban_tile = urban_mask(sr:er, sc:ec,:);
        rural_tile = rural_mask(sr:er, sc:ec,:);
        
        % determine the class label for the tile
        label = setTileLabel(tile_size, factor, ...
        slum_tile, urban_tile, rural_tile);
        
        % save the tile at the location given by the path and label
        saveTile2File(image_tile, extent, tiles_path, label, base_fname, ext);
    end
end

end

