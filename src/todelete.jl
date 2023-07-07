using ExpectationApproximations,LinearAlgebra,ExponentialFamily, AdvancedHMC,ForwardDiff,LogDensityProblems

D = 2
q = MvNormalMeanCovariance(zeros(D), [1.0 0.0; 0.0 1.0] )
ef = convert(ExponentialFamily.KnownExponentialFamilyDistribution, q)
f(x) = exp(-sin(tr(x*x')))
metric = DiagEuclideanMetric(D)
n_samples, n_adapts = 2_0000, 1_00
method = ExpectationApproximations.AHMC(D)

M = [1.0 -0.2; -0.1 4]
DenseEuclideanMetric(M). cholM⁻¹
samplelist = Base.prod(method,q,f)
expectation = ExpectationApproximations.approximate(method,q,f)
# ExpectationApproximations.approximate_distribution(method,ef,f)
AdvancedHMC.getname(metric)

using LogDensityProblems
struct LogTargetDensity
    dim::Tuple{Integer, Integer}
end
using Distributions
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = logpdf(Wishart(3, [1 0; 0 1]), θ)  
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

# Choose parameter dimensionality and initial parameter value
D = (2,2); initial_θ = rand(Wishart(3, [1 0; 0 1]))
ℓπ = LogTargetDensity(D)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 2_000, 1_000

using AdvancedHMC, ForwardDiff
# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, [initial_θ])
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, [initial_θ], n_samples, adaptor, n_adapts; progress=true)
import ExponentialFamily: KnownExponentialFamilyDistribution
using ExponentialFamily
using Distributions
ExponentialFamily.variate_form(Normal(0,1))
