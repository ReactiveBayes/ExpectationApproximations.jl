export ForwardDiffGrad

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

function total_derivative(approximation, f, s::Tuple)
    f_for_grad = (x) -> f(x...)
    gradient_at_s = compute_gradient(approximation.grad, f_for_grad, collect(s))
    return dot(collect(s), gradient_at_s)
end
