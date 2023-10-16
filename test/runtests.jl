using Aqua, CpuId, ReTestItems, ExpectationApproximations

# `ambiguities = false` - there are quite some ambiguities, but these should be normal and should not be encountered under normal circumstances
# `piracy = false` - we extend/add some of the methods to the objects defined in the Distributions.jl
Aqua.test_all(ExpectationApproximations, ambiguities = false, piracy = false)

runtests(ExpectationApproximations, nworkers = cpucores(), nworker_threads = 2, memory_threshold = 1.0)
