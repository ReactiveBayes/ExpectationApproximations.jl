"""
   RescaleArguments(f, scale)

Returns a modified callable `r` , such that new behaviour is defined as `r(x) = f(x * scale)`.

# Fields 
- `f` - base callable object
- `scale` - scaling parameter, may be a number or a tuple (for multiple argument callables)

```jldoctest rescale_arguments
julia> import ExpectationApproximations: RescaleArguments
```

```jldoctest rescale_arguments
julia> r = RescaleArguments(sin, 1.0 / 2.0);

julia> r(1.0) === sin(1.0 / 2.0)
true
```

```jldoctest rescale_arguments
julia> r = RescaleArguments(+, (-1.0, 2.0));

julia> r(1.0, 1.0) === -1.0 + 2.0
true
```

```jldoctest rescale_arguments
julia> r = RescaleArguments(.+, (-1.0, 2.0));

julia> r([ 1.0, 2.0 ], [ -1.0, -2.0 ]) == [ -3.0, -6.0 ]
true
```

See also: [`ExpectationApproximations.RescaleOutput`](@ref), [`ExpectationApproximations.ShiftOutput`](@ref), [`ExpectationApproximations.Decay`](@ref), [`ExpectationApproximations.Displace`](@ref)
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

"""
   RescaleOutput(f, scale)

Returns a modified callable `r` , such that new behaviour is defined as `r(x) = scale * f(x)`.

# Fields 
- `f` - base callable object
- `scale` - scaling parameter, may be a number or a tuple (for callables that return vector or a tuple)

```jldoctest rescale_output
julia> import ExpectationApproximations: RescaleOutput
```

```jldoctest rescale_output
julia> r = RescaleOutput(sin, 1.0 / 2.0);

julia> r(1.0) === (1.0 / 2.0) * sin(1.0)
true
```

```jldoctest rescale_output
julia> r = RescaleOutput(+, -1.0);

julia> r(1.0, 1.0) === -2.0
true
```

```jldoctest rescale_output
julia> r = RescaleOutput(.+, (-1.0, 2.0));

julia> r([ 1.0, 2.0 ], [ -2.0, -3.0 ]) == [ 1.0, -2.0 ]
true
```

See also: [`ExpectationApproximations.RescaleArguments`](@ref), [`ExpectationApproximations.ShiftOutput`](@ref), [`ExpectationApproximations.Decay`](@ref), [`ExpectationApproximations.Displace`](@ref)
"""
struct RescaleOutput{C, S}
    callable::C
    scale::S
end

(rescaled::RescaleOutput)(args...) = rescale_output(rescaled, rescaled.scale, rescaled.callable, args)

rescale_output(::RescaleOutput, scale, callable, args) = scale .* callable(args...)

"""
   ShiftOutput(f, shift)

Returns a modified callable `s` , such that new behaviour is defined as `s(x) = f(x) + shift`.

# Fields 
- `f` - base callable object
- `shift` - shifting parameter, may be a number or a tuple (for callables that return vector or tuple)

```jldoctest shift_output
julia> import ExpectationApproximations: ShiftOutput
```

```jldoctest shift_output
julia> s = ShiftOutput(sin, 1.0 / 2.0);

julia> s(1.0) === sin(1.0) + 1.0 / 2.0
true
```

```jldoctest shift_output
julia> s = ShiftOutput(+, -1.0);

julia> s(1.0, 1.0) === 1.0
true
```

```jldoctest shift_output
julia> s = ShiftOutput(.+, (-1.0, 2.0));

julia> s([ 1.0, 2.0 ], [ -2.0, -3.0 ]) == [ -2.0, 1.0 ]
true
```

See also: [`ExpectationApproximations.RescaleArguments`](@ref), [`ExpectationApproximations.RescaleOutput`](@ref), [`ExpectationApproximations.Decay`](@ref), [`ExpectationApproximations.Displace`](@ref)
"""
struct ShiftOutput{C, S}
    callable::C
    shift::S
end

(shifted::ShiftOutput)(args...) = shift_output(shifted, shifted.shift, shifted.callable, args)

shift_output(::ShiftOutput, shift, callable, args) = callable(args...) .+ shift

"""
    Decay(f, a = nothing)

Returns a modified callable `r` , such that new behaviour is defined as `r(x) = f(x) / x ^ a`. 

# Fields
- `f` - base callable object
- `a` - degree of decaying, in case if `a` is set to `nothing`, then `a = 1` is assumed

```jldoctest decay
julia> import ExpectationApproximations: Decay
```

```jldoctest decay
julia> d = Decay((x) -> x^2);

julia> d(2.0) === 2.0
true
```

```jldoctest decay
julia> d = Decay((x) -> x^2, 3.0);

julia> d(2.0) === 1.0 / 2.0
true
```

See also: [`ExpectationApproximations.RescaleArguments`](@ref), [`ExpectationApproximations.RescaleOutput`], [`ExpectationApproximations.ShiftOutput`](@ref), [`ExpectationApproximations.Displace`](@ref)
"""
struct Decay{C, D}
    callable::C
    degree::D
end

Decay(f::F) where {F <: Function} = Decay(f, nothing)

(decayed::Decay)(x::Real) = decay(decayed, decayed.degree, decayed.callable, x)

decay(::Decay, decay::Nothing, callable, x::Real) = callable(x) / x
decay(::Decay, decay::Real, callable, x::Real) = callable(x) / (x^decay)

"""
    Displace(f, a = nothing)

Returns a modified callable `d` , such that new behaviour is defined as `d(x) = f(x) + a * x`. 

# Fields
- `f` - base callable object
- `a` - scale for displacment, in case if `a` is set to `nothing`, then `a = 1` is assumed

```jldoctest displace
julia> import ExpectationApproximations: Displace
```

```jldoctest displace
julia> d = Displace((x) -> x^2);

julia> d(3.0) === 12.0
true
```

```jldoctest displace
julia> d = Displace((x) -> x^2, 3.0);

julia> d(3.0) === 18.0
true
```

See also: [`ExpectationApproximations.RescaleArguments`](@ref), [`ExpectationApproximations.RescaleOutput`], [`ExpectationApproximations.ShiftOutput`](@ref), [`ExpectationApproximations.Decay`](@ref)
"""
struct Displace{C, S}
    callable::C
    scale::S
end

Displace(f::F) where {F <: Function} = Displace(f, nothing)

(displaced::Displace)(x::Real) = displace(displaced, displaced.scale, displaced.callable, x)

displace(::Displace, scale::Nothing, callable, x::Real) = callable(x) + x
displace(::Displace, scale::Real, callable, x::Real) = callable(x) + scale * x
displace(::Displace, ::Val{S}, callable, x::Real) where {S} = callable(x) + S * x
displace(::Displace, ::Val{1}, callable, x::Real) = callable(x) + x
displace(::Displace, ::Val{-1}, callable, x::Real) = callable(x) - x
