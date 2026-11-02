# Independent products of distributions of mixed kinds (decision D14 of PLAN.md), represented
# by BayesBase's `FactorizedJoint`. The integrand receives a tuple with one value per component.
# A backend supports a product when it supports every component for any integrand, and the
# product is computed as a product rule over the point sets of the components.

"""
    ProductPointSet(sets)

The product rule over a tuple of [`PointSet`](@ref)s: every combination of points, weighted
by the product of their weights. The integrand receives a tuple of points.
"""
struct ProductPointSet{S <: Tuple}
    sets::S
end

Base.length(product::ProductPointSet) = prod(map(length, product.sets))

function expectation(product::ProductPointSet, f)
    sets = product.sets
    indices = CartesianIndices(map(set -> Base.OneTo(length(set)), sets))
    result = _weight(sets, first(indices)) * f(_point(sets, first(indices)))
    for (k, index) in enumerate(indices)
        k == 1 && continue
        result += _weight(sets, index) * f(_point(sets, index))
    end
    return result
end

_point(sets, index) = map((set, i) -> @inbounds(set.points[i]), sets, Tuple(index))
_weight(sets, index) = prod(map((set, i) -> @inbounds(set.weights[i]), sets, Tuple(index)))

# The point set behind a preparation; a product rule needs one for every component
pointset(set::PointSet) = set
pointset(prepared) = throw(ArgumentError(
    "a product rule needs a point set for every component, but the preparation is a $(typeof(prepared))"
))

prepare_expectation(backend, q::FactorizedJoint) = prepare_product(backend, q)

prepare_product(backend, q) = ProductPointSet(map(c -> pointset(prepare_expectation(backend, c)), components(q)))

supports(backend, ::Type, ::Type{FactorizedJoint{T}}) where {T <: Tuple} = supports_all(backend, T)

haslabel(backend, ::Type, ::Type{FactorizedJoint{T}}, label::ExpectationLabel) where {T <: Tuple} =
    haslabel_all(backend, T, label)

# Recursions over the component types, so that inference decides them
supports_all(backend, ::Type{Tuple{}}) = true
supports_all(backend, ::Type{T}) where {T <: Tuple} =
    supports(backend, Any, Base.tuple_type_head(T)) && supports_all(backend, Base.tuple_type_tail(T))

haslabel_all(backend, ::Type{Tuple{H}}, label) where {H} = haslabel(backend, Any, H, label)
haslabel_all(backend, ::Type{T}, label) where {T <: Tuple} = combine_label(
    label, haslabel(backend, Any, Base.tuple_type_head(T), label), haslabel_all(backend, Base.tuple_type_tail(T), label)
)
