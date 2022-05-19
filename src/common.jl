
import Base: iterate, IteratorSize, IteratorEltype, eltype, length, size

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