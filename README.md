# Manifold Interpolation

This project is a realization of the algorithm for manifold interpolation from https://arxiv.org/abs/1508.00674. 

## Generating Points on the canonical relation

The file `shapes.py` can generate ellipses and star-convex polygons and calculate their wavefront sets. In addition, it calculates the wavefront set of the straight line transform in fanbeam coordinates of a given shape. The output format is a list of 6d points, where the first two coordinates are the 'spatial coordinates' (in fanbeam) of the wavefront set of the sinogram, the next two are the spatial coordinates of the original shape, the 5th coordinate is the angle (in degrees) of the wavefront direction in the sinogram, and the 6th is the angle (in degrees) of the wavefront direction in the original shape.

After generating a list of points with `shapes.py`, the points **have** to be rudimentarily deduplicated and sorted, which can be done with:
```bash
cat output.txt | sort -h | uniq > sortedOutput.txt
```

## Mani. Interp. Algorithm

The rust project in the `feffer` directory contains an implementation of the algorithm from the main paper (see page 62). In our case, the way the shapes are generated on a `[0,200]^4 x [0,179]^2` grid, we may take `r=1` and all points will be `1/100`-distant from each other and we will satisfy the requirements for the paper by choosing `\delta < 1`.  