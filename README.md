# Python3 Map-Elites
This repository contains "reference implementations" of:
- CVT-MAP-Elites (Vassiliades, Chatzilygeroudis, Mouret, 2017)
- Multitask-MAP-Elites (Mouret and Maguire, 2020)

CVT-MAP-Elites can be used instead of the standard MAP-Elites described in:
Mouret JB, Clune J. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.

The general philosophy is to provide implementations (1 page of code) that are easy to transform and include in your own research. This means that there is some code redundancy between the algorithms. If you are interested in a more advanced framework:
- Sferes (C++): https://github.com/sferes2/sferes2 
- QDPy (Python) https://pypi.org/project/qdpy/

By default, the evaluations are parallelized on each core (using the multiprocessing package).

## Dependencies

- python3
- numpy
- sklearn (scikit-learn) [for CVT]
- matplotlib (optional, for plotting)

## References:
If you use this code in a scientific paper, please cite:

**Main paper**: Mouret JB, Clune J. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.

**CVT Map-Elites**: Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multi-dimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3.

**Variation operator**: Vassiliades V, Mouret JB. Discovering the Elite Hypervolume by Leveraging Interspecies Correlation. Proc. of GECCO. 2018.

**Multitask-MAP-Elites**: Mouret JB, Maguire G. Quality Diversity for Multi-task Optimization. Proc of GECCO. 2020.


## Basic usage
(you need to have the map_elites module map_elites in your Python path)

```python
import map_elites.cvt as cvt_map_elites

archive = cvt_map_elites.compute(2, 6, rastrigin, n_niches=5000, n_gen=2500)
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
