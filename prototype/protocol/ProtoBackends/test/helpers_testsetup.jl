@testsetup module Helpers

using ProtoBase, ProtoBackends, BenchmarkTools

const P = ProtoBase
const B = ProtoBackends

export P, B, folds_to, allocations

"True if the optimised code of `f` for argument `types` returns the constant `value`."
function folds_to(f, types, value)
    code = only(code_typed(f, types; optimize = true)).first.code
    returns = filter(statement -> statement isa Core.ReturnNode, code)
    length(returns) == 1 || return false
    returned = only(returns).val
    return (returned isa QuoteNode ? returned.value : returned) === value
end

"Allocations of `f(args...)`, measured in a function compiled for the types of `f` and `args`."
allocations(f, args...) = @ballocated $f($args...) samples = 3 evals = 1

end
