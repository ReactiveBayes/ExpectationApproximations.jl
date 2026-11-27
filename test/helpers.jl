@testmodule EAHelpers begin
    using BenchmarkTools: @ballocated
    using ExpectationApproximations, BayesBase

    const EA = ExpectationApproximations
    const P = BayesBase

    export EA, P, folds_to, allocations

    "True if the optimised code of `f` for argument `types` returns the constant `value`."
    function folds_to(f, types, value)
        code = only(code_typed(f, types; optimize = true)).first.code
        returns = filter(statement -> statement isa Core.ReturnNode, code)
        length(returns) == 1 || return false
        returned = only(returns).val
        return (returned isa QuoteNode ? returned.value : returned) === value # quoted on Julia 1.10
    end

    """
    Allocations of `f(args...)`, measured by BenchmarkTools in a function compiled for the types
    of `f` and `args`; `@allocated` in a generic helper counts the boxed result on Julia 1.10.
    """
    allocations(f, args...) = @ballocated $f($args...) samples = 3 evals = 1
end
