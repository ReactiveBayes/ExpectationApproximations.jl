export AbstractApproximationMethod, Expectation, approximate

using Distributions

# TODO revise later on
# struct Compatible end
# struct CompatibilityUnknown end

# iscompatible(method, q) = CompatibilityUnknown()
abstract type AbstractApproximationMethod end
abstract type Kernel end

function adjust_integrand end
function adjust_result end

struct Expectation end

##

as_integrand(f::F) where {F}  = f
as_integrand(f::LogIntegrand) = f
as_integrand(f::Distribution) = LogIntegrand((x) -> logpdf(f, x))

##

approximate(method::AbstractApproximationMethod, f::F) where F    = approximate(method, nothing, f)
approximate(method::AbstractApproximationMethod, q, f::F) where F = approximate(Expectation(), method, q, f)

approximate(::Expectation, method::AbstractApproximationMethod, q, f::F) where F  = approximate_expectation(method, q, as_integrand(f))
approximate(kernel::Kernel, method::AbstractApproximationMethod, q, f::F) where F = approximate_expectation(method, q, adjust_integrand(kernel, as_integrand(f)))

