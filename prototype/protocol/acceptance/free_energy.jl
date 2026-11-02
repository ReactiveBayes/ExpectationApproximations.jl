# End-to-end check of U3 (PLAN.md): the mixture notebook's model, whose likelihood node has no
# average energy, gets a free energy through the rule fallback's average energy, computed with
# the context's expectation backend. Run in the universal-dev environment:
#
#     julia --project=~/Projects/Julia/universal-dev free_energy.jl

using Printf, Statistics

notebook = Module()
redirect_stdout(devnull) do
    Base.include(notebook, joinpath(@__DIR__, "mixture_protocol.jl"))
end

@eval notebook begin
    import MessagePassingRulesBase
    using MessagePassingRulesBase: NodeFunctionRuleFallback

    # The notebook's fallback answers messages; its average energy is the built-in fallback's,
    # -E_q[log f] through the context's backend.
    MessagePassingRulesBase.call_energy_fallback(::ExpectedLogPdfFallback, node, algorithm, ctx, args) =
        MessagePassingRulesBase.call_energy_fallback(NodeFunctionRuleFallback(), node, algorithm, ctx, args)

    function with_free_energy(backend)
        return infer(
            model = laplace_mixture(K = K), data = (y = y,), constraints = laplace_constraints,
            initialization = laplace_initialization, iterations = n_iterations, returnvars = (m = KeepLast(),),
            free_energy = true, options = (rulefallback = ExpectedLogPdfFallback(), context = (expectation = backend,)),
        )
    end
end

passed = true
for (name, backend) in (("EADefault", notebook.EA.EADefault()), ("notebook quadrature", notebook.notebook_router(5)))
    result = Base.invokelatest(notebook.with_free_energy, backend)
    F = result.free_energy
    finite = all(isfinite, F)
    # Laplace's projections need not decrease the free energy exactly; after the first
    # iterations it settles, increasing by no more than rounding.
    settled = all(<=(1.0e-6 * abs(F[end])), diff(F[5:end]))
    # The posterior is the notebook's: the energy changes nothing that is computed.
    same = isapprox(mean(result.posteriors[:m]), mean(notebook.q_m); rtol = 1.0e-10)
    global passed &= finite && settled && same
    @printf(
        "%-20s F[1] = %.4f, F[end] = %.4f, finite %s, non-increasing after 5 %s, posterior unchanged %s\n",
        name, F[1], F[end], finite, settled, same
    )
end

println(passed ? "free energy: PASSED" : "free energy: FAILED")
passed || exit(1)
