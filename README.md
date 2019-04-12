# Python3 Map-Elites (CVT-Map-Elites)
This is a straighforward implementation of CVT-MAP-Elites in Python3.

## Dependencies

- python3
- sklearn (scikit-learn)
- numpy
- matplotlib (for plotting)

## References:
If you use this code in a scientific paper, please cite:

**Main paper**: Mouret JB, Clune J. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.

**CVT Map-Elites**: Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multi-dimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3.

**Variation operator**: Vassiliades V, Mouret JB. Discovering the Elite Hypervolume by Leveraging Interspecies Correlation. Proc. of GECCO 2018.

## Basic usage

```
archive = compute(2, 6, your_function, n_niches=5000, n_gen=2500)
```
Where `2` is the dimensionality of the map, ``6`` is the dimensionality of the genotype, ``n_niches`` is the number of niches, and ``n_gen`` is the number of generation (each generation is 100 evaluations in the default parameters) . You can also pass an optional `params` argument to tune a few parameters. Here are the default values:



```

default_params = \
    {
        "cvt_samples": 25000,
        "batch_size": 100,
        "random_init": 1000,
        "random_init_batch": 100,
        "sigma_iso": 0.01,
        "sigma_line": 0.2,
        "dump_period": 100,
        "parallel": True,
        "cvt_use_cache": True
    }
```
