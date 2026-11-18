"""
    linearize(f, x) -> (A, b)
    linearize(f, x₁, x₂, ...) -> (A, b)

The first-order expansion `f(x) ≈ A x + b` at the point `x`, with derivatives from ForwardDiff,
and `b = f(x) - A x`. Depending on the shapes of the input and the output, `A` is a derivative, a
column, a gradient as a row, or a Jacobian. Several inputs, numbers or vectors, are linearised
jointly, and `A` then acts on their concatenation.
"""
function linearize(f::F, x) where {F}
    A = jacobian(f, x)
    return A, f(x) - A * x
end

function linearize(f::F, x1, x2, xs...) where {F}
    inputs = (x1, x2, xs...)
    sizes = map(length, inputs)
    split(z) = map((input, offset, size) -> piece(input, z, offset, size), inputs, offsets(sizes), sizes)
    return linearize(z -> f(split(z)...), reduce(vcat, map(collect ∘ vec ∘ asarray, inputs)))
end

jacobian(f, x::Real) = ForwardDiff.derivative(f, x)

function jacobian(f, x::AbstractVector)
    f(x) isa Real && return ForwardDiff.gradient(f, x)'
    return ForwardDiff.jacobian(f, x)
end

# The pieces of a concatenation, a number for a number and a vector for a vector
asarray(x::Real) = [x]
asarray(x::AbstractArray) = x
offsets(sizes) = cumsum((0, Base.front(sizes)...))
piece(::Real, z, offset, _) = z[offset + 1]
piece(::AbstractVector, z, offset, size) = z[(offset + 1):(offset + size)]
