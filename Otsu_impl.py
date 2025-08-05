import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 

from skimage.io         import imread_collection, imsave
from skimage.color      import rgb2gray
from skimage.filters    import threshold_otsu
from skimage.morphology import (
    closing, opening, disk,
    remove_small_objects, remove_small_holes
)
from skimage.measure    import label, regionprops

files = '/Frame-*.png'
collection = imread_collection( files )
frames = np.stack( collection, axis=0 ) 
print(f'Loaded {frames.shape[0]} frames of size {frames.shape[1:]}')


out_dir = 'detected_particles'
os.makedirs( out_dir, exist_ok = True )


for i, frame in enumerate( frames ):

    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = rgb2gray( frame )
    else:
        gray = frame  # already 2D

    thr = threshold_otsu( gray )
    mask = gray <= thr


    mask = remove_small_objects( mask, min_size = 50 )
    mask = remove_small_holes( mask, area_threshold = 200 )
    se = disk( 3 )
    mask = closing( opening( mask, se ), se )


    mask = remove_small_objects( mask, min_size=20 )

    # Filter circular particles
    # Using regionprops to filter based on circularity
    lbl = label( mask )
    filtered = np.zeros_like( mask, dtype=bool )

    for region in regionprops( lbl ):
        if region.perimeter == 0:
            continue
        circ = 4 * np.pi * region.area / ( region.perimeter ** 2 )
        if 0.75 <= circ <= 1.0:
            filtered[lbl == region.label] = True


    out_path = os.path.join( out_dir, f'particles_{i:03d}.png' )
    imsave( out_path, filtered.astype(np.uint8) * 255 )

    print(f'Frame {i:03d}: saved inverted mask to {out_path}')

    print(f'Frame {i:03d}: saved {out_path}')

print('All frames processed.')
