function compute_df_mv(approximation::CVI, logp::F, z_s::Real) where {F}
    df_m = compute_derivative(approximation.grad, logp, z_s)
    df_v = compute_second_derivative(approximation.grad, logp, z_s)
    return df_m, df_v / 2
end

function compute_df_mv(approximation::CVI, logp::F, z_s::AbstractVector) where {F}
    df_m = compute_gradient(approximation.grad, logp, z_s)
    df_v = compute_hessian(approximation.grad, logp, z_s)
    return df_m, df_v ./ 2
end

function prod(approximation::CVI, left, dist::GaussianDistributionsFamily)
    rng = something(approximation.rng, Random.GLOBAL_RNG)

    logp = (x) -> logpdf(left, x)

    # Natural parameters of incoming distribution message
    η = naturalparams(dist)
    T = typeof(η)

    # Initial parameters of projected distribution
    λ = naturalparams(dist)

    # Initialize update flag
    hasupdated = false

    for _ in 1:(approximation.n_iterations)
        # create distribution to sample from and sample from it
        q = convert(Distribution, λ)
        z_s = rand(rng, q)

        # compute gradient on mean parameters
        df_m, df_v = compute_df_mv(approximation, logp, z_s)

        # convert mean parameter gradient into natural gradient
        df_μ1 = df_m - 2 * df_v * mean(q)
        df_μ2 = df_v
        ∇f = as_naturalparams(T, df_μ1, df_μ2)

        # compute gradient on natural parameters
        ∇ = λ - η - ∇f

        # perform gradient descent step
        λ_new = as_naturalparams(T, cvi_update!(approximation.opt, λ, ∇))

        # check whether updated natural parameters are proper
        if isproper(λ_new) && enforce_proper_message(approximation.enforce_proper_messages, λ_new, η)
            λ = λ_new
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return convert(Distribution, λ)
end