# Tile a real scn file, load a trained model and run the test.
"""
Created on 09/28/2018

@author: RH
"""

import get_tile
import time
start_time = time.time()
# cut tiles with coordinates in the name (exclude white)

get_tile.tile()

print("--- %s seconds ---" % (time.time() - start_time))

# load 5000 each time until it is done

# output heat map of pos and neg; and output CAM and assemble them to a big graph.

