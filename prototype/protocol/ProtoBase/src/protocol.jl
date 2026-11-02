"""
    expectation(backend, f, q)

The expectation `E_q[f]` of the integrand `f` under the distribution `q`, computed by `backend`.

A backend implements either a closed form, a method of `expectation` for specific types of
`f` and `q`, or a preparation, a method of [`prepare_expectation`](@ref) for specific types of
`q` that returns an object with its own `expectation(prepared, f)`. The fallback goes through
the preparation.

    expectation(prepared, f)

The expectation of `f` from an object returned by [`prepare_expectation`](@ref).
"""
function expectation end

"""
    prepare_expectation(backend, q)

Prepare the expectations of many integrands under `q`: nodes, weights and other work that does
not depend on the integrand are done once. The result is used as `expectation(prepared, f)`,
for any `f`, including integrands whose values are dual numbers.

The default returns a [`Prepared`](@ref), which forwards every call to the backend.
"""
function prepare_expectation end

"""
    Prepared(backend, q)

The default result of [`prepare_expectation`](@ref): no work is done in advance, and
`expectation(prepared, f)` calls `expectation(backend, f, q)`.
"""
struct Prepared{B, Q}
    backend::B
    q::Q
end

prepare_expectation(backend, q) = Prepared(backend, q)

expectation(prepared::Prepared, f) = expectation(prepared.backend, f, prepared.q)

expectation(backend, f, q) = expectation_from(prepare_expectation(backend, q), f)

# A backend with neither a closed form for `(f, q)` nor a preparation for `q` would otherwise
# send `expectation` and the default preparation back and forth forever
expectation_from(prepared, f) = expectation(prepared, f)
expectation_from(prepared::Prepared, f) = throw(NoBackendError(prepared.backend, f, prepared.q))

"""
    NoBackendError(backend, f, q)

Thrown when `backend` cannot compute the expectation of `f` under `q`.
"""
struct NoBackendError{B, F, Q} <: Exception
    backend::B
    f::F
    q::Q
end

function Base.showerror(io::IO, error::NoBackendError)
    print(io, "NoBackendError: `", error.backend, "` cannot compute the expectation of `", error.f)
    print(io, "` under `", error.q, "`")
end

"""
    supports(backend, F::Type, Q::Type)
    supports(backend, f, q)

Whether `backend` computes expectations of integrands of type `F` under distributions of type
`Q`. Backend owners add methods for the type-level form; it is `false` by default. `F = Any`
asks whether every integrand is supported, which is what a numerical method declares with
`supports(::MyBackend, ::Type, ::Type{<:MyFamily}) = true`.
"""
supports(backend, ::Type, ::Type) = false
supports(backend, f, q) = supports(backend, typeof(f), typeof(q))

"""
    which_backend(backend, F::Type, Q::Type)
    which_backend(backend, f, q)

The backend that computes the expectation, or `nothing`. A router returns its choice, any
other backend itself when it supports the pair.
"""
which_backend(backend, ::Type{F}, ::Type{Q}) where {F, Q} = supports(backend, F, Q) ? backend : nothing
which_backend(backend, f, q) = which_backend(backend, typeof(f), typeof(q))
