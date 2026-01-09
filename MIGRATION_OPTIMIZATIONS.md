# Migration Speed Optimizations

## Current Bottlenecks

1. **One function at a time** - We're migrating functions individually
2. **Tests immediately** - Writing tests for each function before moving on
3. **Frequent commits** - Committing after each small feature
4. **Manual repetitive tasks** - Same patterns repeated (optional deps, error handling)

## Recommended Optimizations

### 1. **Batch Module Migration** (Biggest Impact)

Instead of: Migrate function → test → commit → next function

Do: Migrate entire module → batch tests → single commit

**Example**: Migrate all of `geosuite.petro.permeability` in one pass:
- All 6 functions at once
- Add all to `geosmith/primitives/petrophysics.py`
- Write all tests together
- Single commit

**Time savings**: ~70% reduction (from 6 commits to 1)

### 2. **Defer Tests** (Medium Impact)

Instead of: Code → test → next function

Do: Code → code → code → batch tests

**Example**: Migrate 3-5 related functions, then write all tests together

**Time savings**: ~30% reduction (less context switching)

### 3. **Larger Commits** (Small Impact)

Instead of: Commit after each feature (1-2 functions)

Do: Commit after each logical group (5-10 functions or entire module)

**Example**: 
- "Migrate petrophysics permeability calculations" (6 functions)
- "Migrate geomechanics fracture analysis" (4 functions)

**Time savings**: ~10% reduction (less git overhead)

### 4. **Template-Based Migration** (Medium Impact)

Create templates for common patterns:
- Optional dependency handling (scipy, sklearn, numba)
- Error handling patterns
- Function signature standardization

**Example**: Copy-paste template, fill in function logic

**Time savings**: ~20% reduction (less typing)

### 5. **Parallel Discovery** (Small Impact)

Identify all functions to migrate BEFORE starting:
- Scan entire module
- List all public functions
- Group by dependencies
- Migrate in batches

**Time savings**: ~10% reduction (better planning)

## Recommended Workflow

### Phase 1: Bulk Migration (Fast)
```bash
# 1. Scan module
grep -r "^def " geosuite/petro/ | wc -l

# 2. Migrate entire module (code only, no tests)
# Copy all functions to geosmith/primitives/petrophysics.py

# 3. Update exports
# Add to geosmith/primitives/__init__.py

# 4. Single commit
git commit -m "Migrate entire petrophysics module from GeoSuite"
```

### Phase 2: Batch Testing (Thorough)
```bash
# 1. Write all tests for migrated module
# tests/test_petrophysics.py with all test classes

# 2. Run tests
pytest tests/test_petrophysics.py -v

# 3. Fix issues
# Address all test failures in one pass

# 4. Commit
git commit -m "Add comprehensive tests for petrophysics module"
```

### Phase 3: Documentation (Later)
- Add examples after migration complete
- Update docs after code is stable

## Target Metrics

**Current speed**: ~15-20 minutes per function
- Code: 10 min
- Tests: 5 min
- Commit: 1 min

**Optimized speed**: ~5-8 minutes per function
- Code: 3 min (batched)
- Tests: 3 min (batched)
- Commit: 1 min (batched)

**Overall improvement**: 3-4x faster

## Implementation Plan

### Immediate Actions

1. **Migrate by module, not by function**
   - Choose a complete module (e.g., `geosuite.petro.permeability`)
   - Migrate all functions at once
   - Single commit

2. **Defer test creation**
   - Migrate 3-5 related modules first
   - Write tests in batch afterward
   - Better test coverage (see whole picture)

3. **Use helper script**
   - `scripts/batch_migrate_primitives.py` to identify functions
   - Quick scan of dependencies
   - Template generation

### Example: Fast Migration of `geosuite.petro`

**Current approach** (6 separate migrations):
1. `calculate_permeability_kozeny_carman` → test → commit (20 min)
2. `calculate_permeability_timur` → test → commit (20 min)
3. ... (repeat 4 more times)
**Total**: ~120 minutes

**Optimized approach** (1 batch migration):
1. Migrate all 6 functions → single commit (30 min)
2. Write all 6 test classes → single commit (30 min)
3. Fix issues → single commit (20 min)
**Total**: ~80 minutes (33% faster)

### Example: Fast Migration of `geosuite.petro.seismic_processing`

All functions are already migrated! But if starting fresh:
- 5 functions in module
- Current: 5 migrations × 20 min = 100 min
- Optimized: 1 batch migration × 40 min = 40 min
- **60% faster**

## Next Steps

1. **Choose next module to migrate** (e.g., `geosuite.petro.buckles`, `geosuite.petro.lithology`)
2. **Use batch migration workflow**
3. **Measure time savings**
4. **Iterate on process**

