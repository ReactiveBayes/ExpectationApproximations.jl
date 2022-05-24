export AutoDomain, Pdf, LogPdf

import Base: iterate, IteratorSize, IteratorEltype, eltype, length, size, in

""" 
    Points

# Fields 
- `points` : points iterator

Points iterator wrapper for multiple dispatch.

```jldoctest
julia> import ExpectationApproximations: Points

julia> points = Points([ 1, 2, 3 ])
Points{Vector{Int64}}([1, 2, 3])

julia> length(points)
3

julia> sum(points)
6
```

See also: [`ExpectationApproximations.Weights`](@ref)
"""
struct Points{P}
    points::P
end

Base.iterate(wrapper::Points) = iterate(wrapper.points)
Base.iterate(wrapper::Points, state) = iterate(wrapper.points, state)

Base.length(wrapper::Points) = length(wrapper.points)
Base.size(wrapper::Points) = size(wrapper.points)

Base.IteratorSize(wrapper::Points) = IteratorSize(wrapper.points)
Base.IteratorEltype(wrapper::Points) = IteratorEltype(wrapper.points)
Base.eltype(wrapper::Points) = eltype(wrapper.points)

Base.IteratorSize(::Type{Points{P}}) where {P} = IteratorSize(P)
Base.IteratorEltype(::Type{Points{P}}) where {P} = IteratorEltype(P)
Base.eltype(::Type{Points{P}}) where {P} = eltype(P)

""" 
    Weights

# Fields 
- `weights` : weights iterator

Weights iterator wrapper for multiple dispatch.

```jldoctest
julia> import ExpectationApproximations: Weights

julia> points = Weights([ 1, 2, 3 ])
Weights{Vector{Int64}}([1, 2, 3])

julia> length(points)
3

julia> sum(points)
6
```

See also: [`ExpectationApproximations.Points`](@ref)
"""
struct Weights{W}
    weights::W
end

Base.iterate(wrapper::Weights) = iterate(wrapper.weights)
Base.iterate(wrapper::Weights, state) = iterate(wrapper.weights, state)

Base.length(wrapper::Weights) = length(wrapper.weights)
Base.size(wrapper::Weights) = size(wrapper.weights)

Base.IteratorSize(wrapper::Weights) = IteratorSize(wrapper.weights)
Base.IteratorEltype(wrapper::Weights) = IteratorEltype(wrapper.weights)
Base.eltype(wrapper::Weights) = eltype(wrapper.weights)

Base.IteratorSize(::Type{Weights{W}}) where {W} = IteratorSize(W)
Base.IteratorEltype(::Type{Weights{W}}) where {W} = IteratorEltype(W)
Base.eltype(::Type{Weights{W}}) where {W} = eltype(W)

"""
    AutoDomain

Auto domain specification that can be used to create `Pdf` or `LogPdf` structures with "automatic" domain. 
With this domain specification most of the approximation functions would assume that domain of `Pdf` (or `LogPdf`) matches 
domain of other arguments to the approximation function.

See also: [`Pdf`](@ref), [`LogPdf`](@ref)
"""
struct AutoDomain end

Base.in(x, ::AutoDomain) = true

"""
    Pdf

# Fields
- `domain` - domain (e.g. from DomainSets.jl) of probability density function, defaults to `AutoDomain`
- `callable` - base callable object that represents some (potentially unnormalized) probability density function 

```jldoctest pdf
julia> pdf = Pdf((x) -> exp(-(x - 1.0)^2));

julia> pdf(1.0)
1.0
```

See also: [`LogPdf`](@ref), [`AutoDomain`](@ref)
"""
struct Pdf{D, F}
    domain   :: D
    callable :: F
end

Pdf(f::F) where { F <: Function } = Pdf(AutoDomain(), f)

function (pdf::Pdf)(x)
    @assert x ∈ pdf.domain "Value x = $x does not belong to the domain = $(pdf.domain)"
    return pdf.callable(x)
end

"""
    LogPdf

# Fields
- `domain` - domain (e.g. from DomainSets.jl) of log probability density function, defaults to `AutoDomain`
- `callable` - base callable object that represents some (potentially unnormalized) log probability density function 

```jldoctest logpdf
julia> logpdf = Pdf((x) -> -(x - 1.0)^2);

julia> logpdf(1.0)
0.0
```

See also: [`Pdf`](@ref), [`AutoDomain`](@ref)
"""
struct LogPdf{D, F}
    domain   :: D
    callable :: F
end

LogPdf(f::F) where { F <: Function } = LogPdf(AutoDomain(), f)

function (logpdf::LogPdf)(x)
    @assert x ∈ logpdf.domain "Value x = $x does not belong to the domain = $(logpdf.domain)"
    return logpdf.callable(x)
end

"""
   RescaleArguments

# Fields 
- `callable` - base callable (single argument) object
- `scale` - scaling parameter, may be a number or a tuple (for multiple argument callables)

```jldoctest rescale
julia> import ExpectationApproximations: RescaleArguments
```

```jldoctest rescale
julia> f = RescaleArguments(sin, 1.0 / 2.0);

julia> f(1.0) === sin(1.0 / 2.0)
true
```

```jldoctest rescale
julia> f = RescaleArguments(+, (-1.0, 2.0));

julia> f(1.0, 1.0) === -1.0 + 2.0
true
```

```jldoctest rescale
julia> f = RescaleArguments(.+, (-1.0, 2.0));

julia> f([ 1.0, 2.0 ], [ -1.0, -2.0 ]) == [ -3.0, -6.0 ]
true
```

Returns a callable object that is equivalent to (x...) -> callable((scale .* x)...).
"""
struct RescaleArguments{C, S}
    callable::C
    scale::S
end

(rescaled::RescaleArguments)(args...) = rescale_arguments(rescaled, rescaled.scale, rescaled.callable, args)

function rescale_arguments(::RescaleArguments, scale::Number, callable, args) 
    return callable(map(Base.Broadcast.BroadcastFunction(Base.Fix1(*, scale)), args)...)
end

function rescale_arguments(::RescaleArguments, scale::Tuple, callable, args) 
    @assert length(scale) === length(args) "Length of the scaling parameter $(scale) does not equal to the number of arguments."
    return callable(map(.*, scale, args)...)
end