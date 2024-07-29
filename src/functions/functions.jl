"""
    inplace(f::Function)

Returns a function that performs inplace operation on the first argument of the returned function.
"""
function inplace(f::Function)
    return (y, x...) -> begin
        y .= f(x...)
        return y
    end
end

"""
    FunctionWithMemory{F <: Function, M}

A function with memory.	Calling this function is equivalent to calling `f.f(x..., f.memory)`.
"""
struct FunctionWithMemory{F <: Function, M} <: Function
    f::F
    memory::M
end
(f::FunctionWithMemory)(x...) = f.f(x..., f.memory)
