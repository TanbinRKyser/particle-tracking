import os
import numpy as np
from skimage.io       import imread_collection
from skimage.color    import rgb2gray
from skimage.feature  import blob_log
from scipy.optimize   import linear_sum_assignment
import matplotlib

# Ensure matplotlib does not use an interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# ─── CONFIG ────────────────────────────────────────────
mask_pattern = '~/particles_*.png' 
raw_pattern  = '~/Frame-*.png' 
out_dir      = 'kalman_pretty'
os.makedirs( out_dir, exist_ok = True ) 

# LoG detection params (same as before)
## num_sigma = 8 gives a good balance between sensitivity and noise reduction
### max_sigma = 9 is the maximum scale for the LoG filter
### min_sigma = 5 is the minimum scale for the LoG filter
### threshold = 0.1 is the detection threshold
min_sigma, max_sigma, num_sigma, threshold = 5, 9, 8, 0.1

# Kalman matrices
## ── KALMAN FILTER CONFIG ────────────────────────────
# State vector: [x, y, vx, vy]  
# where (x, y) is the position and (vx, vy) is the velocity
# State dimension is 4 (2D position + 2D velocity)
# Measurement vector: [x, y] (only position)    
# Measurement dimension is 2
# ────────────────────────────────────────────────
# Time step
dt = 1.0

# State transition matrix
F = np.array([ 
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1,  0],
                [0, 0, 0,  1]
            ])

# Measurement matrix
H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])

## Process noise covariance matrix
Q = np.eye( 4 ) * 0.01

## Measurement noise covariance matrix
R = np.eye( 2 ) * 1.0

# Maximum distance for association and maximum misses before track deletion
max_dist, max_misses = 15.0, 2

# ─── TRACK CLASS ───────────────────────────────────────
# The Track class implements a simple Kalman filter track.
# It maintains the state vector, covariance matrix, and a trace of positions.
class Track:
    """A simple Kalman filter track."""
    def __init__( self, pos, tid ):
        self.x = np.array( [pos[0], pos[1], 0., 0.] )
        self.P = np.eye( 4 )
        self.id = tid
        self.trace = [ pos.copy() ]
        self.misses = 0

    """Predict the next position using the Kalman filter."""
    # The predict method updates the state vector and covariance matrix 
    # based on the state transition matrix and process noise.
    # It returns the predicted position (x, y).
    # The update method adjusts the state vector and covariance matrix
    # based on the measurement (z) and measurement noise.
    # The miss method appends the current position to the trace and increments the miss count.
    # The trace keeps track of the positions over time.
    # The misses attribute counts how many consecutive frames the track has not been updated.
    def predict( self ):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[ :2 ]

    """Update the track with a new measurement."""
    # The update method takes a measurement (z) and adjusts the state vector and covariance matrix
    # using the Kalman gain. It appends the new position to the trace and resets the miss count.
    # The Kalman gain is computed based on the measurement noise and the predicted covariance.
    # It uses the measurement matrix (H) to relate the state vector to the measurement.
    # The trace is updated with the new position, and the misses count is reset.
    def update( self, z ):
        y = z - ( H @ self.x )
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv( S )
        self.x = self.x + K @ y
        self.P = ( np.eye( 4 ) - K @ H ) @ self.P
        self.trace.append( self.x[ :2 ].copy() )
        self.misses = 0

    """Mark the track as missed."""
    # The miss method appends the current position to the trace and increments the misses count.
    # This method is called when a track does not receive an update for a certain number of frames.
    # It helps in tracking the continuity of the track and deciding when to delete it.
    def miss( self ):
        self.trace.append( self.x[ :2 ].copy() )
        self.misses += 1

# ─── LOAD DATA ─────────────────────────────────────────
masks = np.stack( imread_collection( mask_pattern ), axis=0 ).astype( float ) / 255
raws  = np.stack( imread_collection( raw_pattern ), axis=0 )
T = len( masks )

# prepare a colormap
cmap = cm.get_cmap( 'tab20', 256 )

tracks, next_id = [], 0

# ─── TRACKING LOOP ────────────────────────────────────
# Loop through each frame and perform tracking
for t in range(T):
    mask = masks[ t ]
    raw  = raws[ t ]
    gray = rgb2gray( raw ) if raw.ndim == 3 else raw

    # detect on the mask 
    # Using LoG to detect blobs in the mask
    # The blob_log function detects blobs in the mask using the Laplacian of Gaussian method
    # It returns an array of detected blobs with their coordinates and scales.
    # The parameters min_sigma, max_sigma, num_sigma, and threshold control the detection sensitivity
    blobs = blob_log( mask,
                    min_sigma = min_sigma,
                    max_sigma = max_sigma,
                    num_sigma = num_sigma,
                    threshold = threshold
                    )
    dets = blobs[ :, :2 ]  # (y, x)

    # predict
    # Predict the next position for each track using the Kalman filter
    # The predict method is called for each track to get the predicted position.
    # The predicted positions are stored in the preds array.
    preds = np.array( [ tr.predict() for tr in tracks ] ) if tracks else np.empty( ( 0, 2 ) )

    # associate
    # Associate detected blobs with existing tracks using the Hungarian algorithm
    # The linear_sum_assignment function is used to find the optimal assignment of detections to tracks
    # based on the distance between predicted positions and detected positions.
    # The cost matrix is computed as the Euclidean distance between predicted positions and detections.
    # The assignments are made based on the minimum cost, and unmatched tracks and detections are handled.
    # The assigned_t and assigned_d sets keep track of which tracks and detections were matched
    assigned_t, assigned_d = set(), set()
    
    if len( tracks ) and len( dets ):
        
        D = np.linalg.norm( preds[ :, None, : ] - dets[ None, :, : ], axis = 2 )
        
        cost = np.where( D > max_dist, 1e6, D )
        
        row, col = linear_sum_assignment( cost ) ### Hungarian algorithm for assignment
        
        for r,c in zip( row, col ):
            if cost[ r, c ] < 1e6:
                tracks[ r ].update( dets[ c ] )
                assigned_t.add( r ); assigned_d.add( c ) #### assigned_t = tracks, assigned_d = detections

    # unmatched tracks → miss
    # For tracks that were not matched with any detection, the miss method is called
    # This increments the miss count and appends the current position to the trace.
    for i_tr in set( range( len( tracks ) ) ) - assigned_t:
        tracks[ i_tr ].miss()

    # unmatched detections → new tracks
    # For detections that were not matched with any track, new Track objects are created
    # and added to the tracks list. Each new track is initialized with the detection position.
    # The next_id is incremented for each new track created.
    for i_det in set( range( len( dets ) ) ) - assigned_d:
        tracks.append( Track( dets[ i_det ], next_id ) )
        next_id += 1

    # Remove tracks that have too many misses
    # The tracks list is filtered to keep only those tracks that have misses less than or equal to max_misses.
    # This ensures that tracks that have been inactive for too long are removed.
    tracks = [ tr for tr in tracks if tr.misses <= max_misses ] # remove tracks with too many misses



    ## DEBUGGING: print track info
    # ─── PLOT ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6,6), facecolor='black')
    ax.imshow(gray, cmap='gray', interpolation='nearest')
    ax.set_axis_off()

    for tr in tracks:
        pts = np.array(tr.trace)
        color = cmap(tr.id % 256)
        ax.plot(pts[:,1], pts[:,0], '-', color=color, linewidth=1.2, alpha=0.8)
        # endpoint
        y, x = pts[-1]
        ax.scatter([x],[y], s=30, c=[color], edgecolors='white', linewidths=0.5)
        ax.text(x+2, y-2, str(tr.id),
                color=color, fontsize=6, weight='bold')

    out_path = os.path.join(out_dir, f'track_{t:03d}.png')
    fig.savefig(out_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"[{t+1}/{T}] saved {out_path}")

print("All done!")
