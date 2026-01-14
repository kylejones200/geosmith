# GeoSmith Performance Guide

This document provides performance benchmarks and optimization guidelines for GeoSmith.

## Overview

GeoSmith is designed for performance with:
- **Numba acceleration** for critical computations (variogram, distance calculations)
- **Efficient algorithms** optimized for large datasets
- **Optional parallelization** for computationally intensive operations
- **Memory-efficient** data structures

## Performance Benchmarks

### Variogram Computation

Variogram computation is accelerated with Numba JIT compilation for datasets with >100 samples.

| Sample Size | Compute Time | Fit Time | Total Time |
|------------|--------------|----------|------------|
| 100        | ~5-10 ms     | ~1-2 ms  | ~6-12 ms   |
| 500        | ~20-40 ms    | ~2-3 ms  | ~22-43 ms  |
| 1,000      | ~50-100 ms   | ~3-5 ms  | ~53-105 ms |
| 5,000      | ~500-1000 ms | ~10-20 ms| ~510-1020 ms|

**Performance Tips:**
- Use `n_lags=15` for most applications (default)
- Reduce `n_lags` for very large datasets if precision can be sacrificed
- Numba acceleration is automatic for datasets >100 samples

### Kriging Performance

Kriging performance scales with the number of samples and targets.

#### Ordinary Kriging

| Samples | Targets | Fit Time | Predict Time | Throughput |
|---------|---------|----------|--------------|------------|
| 50      | 500     | ~5-10 ms | ~20-40 ms    | ~12,000-25,000 pred/s |
| 200     | 2,000   | ~20-50 ms| ~100-200 ms  | ~10,000-20,000 pred/s |
| 500     | 5,000   | ~100-200 ms| ~500-1000 ms | ~5,000-10,000 pred/s |

**Performance Tips:**
- Fit once, predict many times (kriging system is cached)
- For very large grids, consider batch processing
- Use Simple Kriging if mean is known (slightly faster)

#### Co-Kriging

Co-Kriging is more computationally intensive due to larger covariance matrices.

| Primary | Secondary | Targets | Fit Time | Predict Time |
|---------|-----------|---------|----------|--------------|
| 30      | 50        | 500     | ~30-60 ms| ~50-100 ms   |
| 100     | 200       | 2,000   | ~200-400 ms| ~400-800 ms |

**Performance Tips:**
- Co-Kriging is beneficial when secondary variable is much denser
- Consider using Ordinary Kriging if cross-correlation is weak

### Sequential Gaussian Simulation (SGS)

SGS performance depends on number of samples, targets, and realizations.

| Samples | Targets | Realizations | Total Time | Time/Realization |
|---------|---------|--------------|-----------|------------------|
| 30      | 500     | 5            | ~2-5 s    | ~0.4-1.0 s      |
| 50      | 1,000   | 10           | ~5-10 s   | ~0.5-1.0 s      |
| 100     | 2,000   | 10           | ~15-30 s  | ~1.5-3.0 s      |

**Performance Tips:**
- Realizations are independent - can be parallelized
- Use fewer realizations for initial exploration
- Consider using kriging for point estimates, SGS for uncertainty

## Optimization Guidelines

### 1. Dataset Size

**Small datasets (<100 samples):**
- All operations are fast
- No special optimization needed

**Medium datasets (100-1,000 samples):**
- Numba acceleration automatically enabled
- Consider reducing `n_lags` for variograms if needed

**Large datasets (>1,000 samples):**
- Use appropriate `n_lags` (15-20 is usually sufficient)
- Consider data quality filtering to remove outliers/duplicates
- For kriging, fit once and predict in batches

### 2. Grid Size

**Small grids (<1,000 blocks):**
- Direct kriging is fast
- No optimization needed

**Medium grids (1,000-10,000 blocks):**
- Consider using block model sub-sampling for initial exploration
- Use Simple Kriging if mean is known

**Large grids (>10,000 blocks):**
- Process in batches or tiles
- Consider using IDW for initial estimates, kriging for refinement
- Use rotated grids to align with geological structures (reduces search radius)

### 3. Memory Management

- **PointSet objects** are memory-efficient (numpy arrays)
- **Kriging systems** are cached after fitting (reuse when possible)
- **SGS realizations** can be generated one at a time to save memory

### 4. Parallelization

Currently available:
- `calculate_overburden_stress_parallel` for well arrays
- Numba parallel execution for variogram computation

Future enhancements:
- Parallel kriging predictions
- Parallel SGS realizations
- Parallel block model generation

## Running Benchmarks

To run performance benchmarks:

```bash
# Run all benchmarks
python -m benchmarks.run_all_benchmarks

# Run specific benchmark
python -m benchmarks.benchmark_kriging
python -m benchmarks.benchmark_variogram
python -m benchmarks.benchmark_simulation
```

Results are saved to `benchmarks/results.json` and printed to console.

## Performance Regression Testing

Benchmarks are included in the test suite to detect performance regressions:

```bash
pytest benchmarks/ -v
```

## Hardware Considerations

**CPU:**
- Single-threaded performance is most important (Numba benefits)
- Multi-core helps with parallel operations (when available)

**Memory:**
- 8 GB RAM sufficient for most workflows
- 16+ GB recommended for large block models (>100,000 blocks)

**Storage:**
- SSD recommended for I/O operations (GSLIB, GRDECL)

## Best Practices

1. **Profile first**: Use `cProfile` or `line_profiler` to identify bottlenecks
2. **Cache fits**: Fit kriging models once, predict many times
3. **Batch processing**: Process large grids in batches
4. **Data quality**: Filter outliers/duplicates before processing
5. **Appropriate models**: Use simpler models (Simple Kriging) when possible
6. **Grid optimization**: Use rotated grids and sub-blocking strategically

## Future Performance Enhancements

Planned improvements:
- Parallel kriging predictions
- GPU acceleration for large grids
- Distributed computing support (Dask)
- Memory-mapped file I/O for large datasets
- Incremental variogram computation

## References

- [Numba Documentation](https://numba.pydata.org/)
- [Scipy Performance Guide](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/basics.performance.html)

