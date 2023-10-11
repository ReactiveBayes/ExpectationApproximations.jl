function init_model(x::AbstractArray)
    return (params = (x,),)
end

hybridlinearize(vector::AbstractVector) = vector
hybridlinearize(matrix::AbstractMatrix) = eachcol(matrix)

# We perform the check in case if the `enforce_proper_messages` setting is set to `Val{true}`
function enforce_proper_message(
    ::Val{true}, ef_numerator::ExponentialFamily.ExponentialFamilyDistribution{T}, ef_denominator::ExponentialFamily.ExponentialFamilyDistribution{T}
) where {T}
    num_param = ExponentialFamily.getnaturalparameters(ef_numerator)
    denom_param = ExponentialFamily.getnaturalparameters(ef_denominator)
    return ExponentialFamily.isproper(ExponentialFamily.ExponentialFamilyDistribution(T, num_param - denom_param, ExponentialFamily.getconditioner(ef_numerator), nothing))
end

# # We skip the check in case if the `enforce_proper_messages` setting is set to `Val{false}`
enforce_proper_message(::Val{false}, ef_1::ExponentialFamily.ExponentialFamilyDistribution{T}, ef_2::ExponentialFamily.ExponentialFamilyDistribution{T}) where {T} = true
