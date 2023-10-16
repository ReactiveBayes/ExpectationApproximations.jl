using StatsFuns: logistic
using StatsFuns: softmax, softmax!
using LoopVectorization
using SpecialFunctions: gamma, loggamma

import LinearAlgebra
import Base: show, maximum
import Base: convert, promote_rule

"""
    diageye(::Type{T}, n::Int)

An alias for the `Matrix{T}(I, n, n)`. Returns a matrix of size `n x n` with ones (of type `T`) on the diagonal and zeros everywhere else.
"""
diageye(::Type{T}, n::Int) where {T <: Real} = Matrix{T}(I, n, n)

"""
    diageye(n::Int)

An alias for the `Matrix{Float64}(I, n, n)`. Returns a matrix of size `n x n` with ones (of type `Float64`) on the diagonal and zeros everywhere else.
"""
diageye(n::Int) = diageye(Float64, n)