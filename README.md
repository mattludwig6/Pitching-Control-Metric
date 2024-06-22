These are the main files used to compute a pitcher's control in this way.
generalizedClusterEM_AllPitches: create rankings for all pitch types for qualified pitchers, compute control metric for specific (pitcher, pitch, handedness of batter)
pNPL-45: initial attempt at finding distribution and control metrics for specific cases with low n (i.e. fastballs in 3-2 counts against "power hitters")
pitch_optimization: basic simulation environment referenced in appendix

NOTE: the data was initially retrieved using pybaseball (https://github.com/jldbc/pybaseball). None of the above files get new data within the functions. To run these methods with new data either download a new .csv and enter the filename or set up a call within the method
