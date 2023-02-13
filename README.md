# Sinogram to WF of original picture

We want to be able to do the following: given a 2d picture (sinogram) that comes from integrating an object along geodesics determine the wavefront set of the original object. So in some sense we're just trying to recover the geometry.

Intermediate steps:
1. Given a sinogram, determine its wavefront set. This is to be done with [DeNSE](https://github.com/arsenal9971/DeNSE). (We might be able to use the pre-trained version, but can generate training data ourselves. See `DeNSE re-training`.)
2. We have two options:
    1. Use the fact that (WFset of sino, WFset of orig) is a manifold, generate points on this manifold (using `shapes.py`), and interpolate this manifold. See section `Manifold Interpolation` for this. 
    2. Use NN to find the map (sinogram, WFset of sino) -> WFset of orig. 

## DeNSE re-training

We will re-train dense on images of sinograms and their WFset by constructing these in the euclidean geometry, see `shapes.py`, this is to say:
1. Generate shape using `shapes.py`.
2. Calculate the sinogram (I guess just scikit or something)
3. Using `shapes.py` calculate the WFset of the original shape. REMARK: THERE IS CURRENTLY A BUG IN THIS STEP: run `fullEllipseRoutineTimer` and notice that the Sinogram WFset has a kink of 90 degrees somewhere where that shouldn't happen.

## Realization of 2.ii.

We generate training data here as follows:

1. Generate shapes with WFset (`shapes.py`).
2. Calculate their sinogram in some geometry (using some package that I've been told exists but I don't know it).
3. Put the result from 2. into DeNSE to get the WFset of this sinogram. 

Then hopefully some ML magic will get this to work.

## Manifold Interpolation

This part is a realization of the algorithm for manifold interpolation from https://arxiv.org/abs/1508.00674. 

NOTE: IN THE NON-EUCLIDEAN CASE WE WILL NOT KNOW WHICH WAVEFRONT DIRECTION CAME FROM WHICH POINT IN THE INPUT. THIS MEANS WE CANNOT ACTUALLY GENERATE ARBITRARY POINTS ON THIS MANIFOLD IN NON-EUCLIDEAN SPACE. So in some sense this entire thing is useless.

### Generating Points on the canonical relation

The file `shapes.py` can generate ellipses and star-convex polygons and calculate their wavefront sets. In addition, it calculates the wavefront set of the straight line transform in fanbeam coordinates of a given shape. The output format is a list of 6d points, where the first two coordinates are the 'spatial coordinates' (in fanbeam) of the wavefront set of the sinogram, the next two are the spatial coordinates of the original shape, the 5th coordinate is the angle (in degrees) of the wavefront direction in the sinogram, and the 6th is the angle (in degrees) of the wavefront direction in the original shape.

After generating a list of points with `shapes.py`, the points **have** to be rudimentarily deduplicated and sorted, which can be done with:
```bash
cat output.txt | sort -h | uniq > sortedOutput.txt
```

### Mani. Interp. Algorithm

The rust project in the `feffer` directory contains an implementation of the algorithm from the main paper (see page 62). In our case, the way the shapes are generated on a `[0,200]^4 x [0,179]^2` grid, we may take `r=1` and all points will be `1/100`-distant from each other and we will satisfy the requirements for the paper by choosing `\delta < 1`.  