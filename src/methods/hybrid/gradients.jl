struct ForwardDiffGrad end

compute_derivative(::ForwardDiffGrad, f::F, value::Real) where {F}       = ForwardDiff.derivative(f, value)
compute_gradient(::ForwardDiffGrad, f::F, vec::AbstractVector) where {F} = ForwardDiff.gradient(f, vec)
compute_hessian(::ForwardDiffGrad, f::F, vec::AbstractVector) where {F}  = ForwardDiff.hessian(f, vec)

function compute_second_derivative(grad::G, logp::F, z_s::Real) where {G, F}
    first_derivative = (x) -> compute_derivative(grad, logp, x)
    return compute_derivative(grad, first_derivative, z_s)
end

compute_gradient(grad::ForwardDiffGrad, f::F, value::Real) where {F}     = compute_derivative(grad, f, value)

function total_derivative(grad::ForwardDiffGrad, f, s::Real)
    return s * compute_derivative(grad, f, s)
end

using Requires

function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        function cvi_update!(opt::Flux.Optimise.AbstractOptimiser, λ::NaturalParameters, ∇::NaturalParameters)
            return Flux.Optimise.update!(opt, vec(λ), vec(∇))
        end
    end

    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        export ZygoteGrad

        struct ZygoteGrad end

        compute_gradient(::ZygoteGrad, f::F, vec::AbstractVector) where {F} = Zygote.gradient(f, vec)[1]
        compute_hessian(::ZygoteGrad, f::F, vec::AbstractVector) where {F}  = Zygote.hessian(f, vec)
        compute_derivative(::ZygoteGrad, f::F, value::Real) where {F}       = Zygote.gradient(f, value)[1]
    end

    @require DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5" begin
        function compute_df_mv(::CVI{R, O, ForwardDiffGrad}, logp::F, vec::AbstractVector) where {R, O, F}
            result = DiffResults.HessianResult(vec)
            result = ForwardDiff.hessian!(result, logp, vec)
            return DiffResults.gradient(result), DiffResults.hessian(result) ./ 2
        end
    end
end