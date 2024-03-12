export GenConstrainedUnscented, GenCUT, GenConstrainedUnscentedTransform
using BlockArrays, BlockDiagonals
using  KWLinalg,TensorCore

const DEFAULT_Θ = 0.5
"""
The `GenConstrainedUnscented` structure defines the approximation method of Generalized Constrained Unscented Transform.
# Arguments
- `lb`: lower bound on the sigma points #1
- `ub`: upper bound on the sigma points #2
lb = (1,) will imply univariate 
ub = (3,)
lb = ([2,1], 3, 4) will imply first variable is 2 two dimensional, 2nd and 3rd variable are univariate
ub = ([3,5], 5, 8)

The `GenConstrainedUnscented` structure with default parameters can be constructed as `GenConstrainedUnscented()`.
"""
struct GenConstrainedUnscented{Tout<:VariateForm, Tin <:VariateForm, T, R} <: AbstractApproximationMethod
    lb::T
    ub::T
    inputdim::Int
    outputdim::Int
    θ::R
    cache::GenUnscentedCache
end

function GenConstrainedUnscented{Tout, Tin}(lb::T, ub::T, id::Int, od::Int, θ::R = DEFAULT_Θ)  where {Tout <:VariateForm, Tin<:VariateForm, T, R}
    @assert size.(lb) == size.(ub) "Upper bound $(ub) and lower bound $(lb) are not of same size"
    @assert sum(length.(lb)) == id "Sum of the length of upper bound $(ub) and lower bound $(lb) elemets should add up to inputdim"
    @assert all(map(d -> all(first(d) .< getindex(d,2)), zip(lb,ub)))  "Upper bound elements $(ub) should be larger than lower bound elements $(lb) "
    @assert 0 < θ < 1 "Free parameter θ=$(θ) should be in between 0 and 1"
  
    return GenConstrainedUnscented{Tout, Tin, T, R}(lb, ub, id, od, θ, GenUnscentedCache())
end

GenConstrainedUnscented(lb::T,ub::T,θ::R = DEFAULT_Θ) where {T, R} = GenConstrainedUnscented{Univariate,Univariate}(lb,ub,1,1, θ)

"""An alias for the [`GenConstrainedUnscented`](@ref) approximation method."""
const GenCUT = GenConstrainedUnscented

"""An alias for the [`GenUnscented`](@ref) approximation method."""
const GenConstrainedUnscentedTransform = GenConstrainedUnscented

getlowerbound(approximation::GenConstrainedUnscented) = approximation.lb
getupperbound(approximation::GenConstrainedUnscented) = approximation.ub
getθ(approximation::GenConstrainedUnscented)          = approximation.θ


## Univariate sigma points and weights
function sigma_points_weights!(approximation::GenConstrainedUnscented, m::Real, V::Real, S::Real, K::Real)
    L     = sqrt(V)
    invL3 = inv(L^3) 
    minK = L^4*(S*inv(L^3))^2
    K = (K <= minK) ? 1.001minK : K
    u     = (1/2)*(-S*invL3 + (1/V)*sqrt(4*K - 3*(S^2)/V))
    v     = u + S*invL3 

    lb = first(getlowerbound(approximation))
    ub = first(getupperbound(approximation))
    θ  = getθ(approximation)
   
    cache   = approximation.cache
    weights = getcache(cache, (:weights, 3)) 
    points  = getcache(cache, (:points, (1, 3)))
    @views begin
        points[1,1] = m
        points[1,2] = m - u*L
        points[1, 3] = m + v*L
        weights[3] = inv(v*(u+v))
        weights[2] = (v/u)*weights[3]
        weights[1] = 1 - weights[3] - weights[2]
    end
    ##lowerbound
    u = @views points[1,2] < lb ? θ*abs((m - lb)/L) : u
    vnew = @views points[1,3] < lb ? θ*abs((lb - m)/L) : v
    vnew = vnew == v ? u + S*invL3  : vnew
    @views begin
        points[1,1] = m
        points[1,2] = m - u*L
        points[1, 3] = m + vnew*L
    end
    ##upperbound
    u = @views points[1,2] > ub ? θ*abs((m - ub)/L) : u
    vnew2 = @views points[1,3] > ub ? θ*abs((ub - m)/L) : vnew
    vnew2 = vnew2 == vnew ? u + S*invL3 : vnew2
    @views begin
        points[1,1] = m
        points[1,2] = m - u*L
        points[1,3] = m + vnew2*L
    end
    
    auxiliary = inv(vnew2*(u+vnew2))
    @views begin
        weights[3] = auxiliary
        weights[2] = (vnew2/u)*auxiliary
        weights[1] = 1-auxiliary*(vnew2/u +1)
    end
    return nothing
end


#Multivariate sigma points and weights
function form_generalized_constrained_sigma_points!(cache::GenUnscentedCache, m::AbstractVector, lb::T, ub::T, θ::R = DEFAULT_Θ) where {T, R}
    dim = length(m)
    form_generalized_sigma_points!(cache, m)

    lowerbound = reduce(vcat, lb)
    upperbound = reduce(vcat, ub)

    points = getcache(cache, (:points, (dim, 2*dim+1)))
  
    u = getcache(cache, (:u, dim))
    v = getcache(cache, (:v, dim))
    L = getcache(cache, (:mul_A_B_At!_ABAt, (dim,dim)))
    vnew = getcache(cache, (:vnew, dim))
    vnew2 = getcache(cache, (:vnew2, dim))
    tmp = getcache(cache, (:tmp, dim))
    comparator = Vector{Bool}(undef, 2dim)

    ## lower bound
    @inbounds for i=1:dim
        @views u[i] = any(map!((x,y) -> isless(x , y), comparator[1:dim], points[:, i+1], lowerbound) ) ?  minimum(map!((x,y,z) -> θ*abs((x - y) / z), tmp, m , lowerbound, @views L[:,i])) : u[i]
        @views vnew[i] = any(map!((x,y) -> isless(x, y),comparator[dim+1:end], points[:, dim+i+1] ,lowerbound) ) ?  minimum(map!((x,y,z) -> θ*abs(x - y)/z, tmp, lowerbound, m, @views L[:,i])) : v[i]
    end
    L3S = getcache(cache, (:L3S, dim))
    vnew = vnew == v ? vnew = u + L3S : vnew

    @inbounds for (u_scal,v_scal, L_vec, i) in zip(u, vnew, eachcol(L), 1:dim)
        map!((L_scal, j) -> m[j] - u_scal * L_scal, view(points, :, i+1), L_vec, 1:dim)
        map!((L_scal, j) -> m[j] + v_scal * L_scal, view(points, :, dim+i+1), L_vec, 1:dim)
    end
    
    ## upper bound
    @inbounds for i=1:dim
        @views u[i] = any(map!((x,y) -> x  > y,comparator[1:dim], points[:, i+1] , upperbound )) ? minimum(map!((x,y,z) ->  θ*abs((x -y)/z), tmp, m, upperbound, @views L[:,i])) : u[i]
        @views vnew2[i] = any(map!((x,y) -> x > y,comparator[dim+1:end],  points[:, dim+i+1] , upperbound)) ? minimum(map!((x,y,z) -> θ*abs((y -x)/z),tmp, m, upperbound, @views L[:,i])) : vnew[i]
    end
    vnew2 = vnew2 == vnew ? vnew2 = u + L3S : vnew2

    @inbounds for (u_scal,v_scal, L_vec, i) in zip(u, vnew2, eachcol(L), 1:dim)
        map!((L_scal, j) -> m[j] - u_scal * L_scal, view(points, :, i+1), L_vec, 1:dim)
        map!((L_scal, j) -> m[j] + v_scal * L_scal, view(points, :, dim+i+1), L_vec, 1:dim)
    end
    
    weights = getcache(cache, (:weights, 2*dim+1))
    map!((x,y) -> inv(y)/(y+x), view(weights, dim+2:2dim+1) ,u, v)
    map!((x,y, z) -> z*x/y, view(weights, 2:dim+1),v, u, view(weights, dim+2:2dim+1))
    @views weights[1] = 1 - sum(@views weights[2:end])

    return nothing
end

function sigma_points_weights!(approximation::GenConstrainedUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector)
    gen_unscented_cache = approximation.cache 
    prepare_cache!(gen_unscented_cache, V, S, K)
    form_generalized_constrained_sigma_points!(gen_unscented_cache, m, getlowerbound(approximation),getupperbound(approximation), getθ(approximation))

    return nothing
end

# function approximate_expectation(method::GenConstrainedUnscented, q, f::F) where {F}
#     m = mean(q)
#     V = _cov(q)
#     S = skewness(q)
#     K = kurtosis(q, false)
#     (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)
#     return sum(weights .* f.(sigma_points))
# end

# function approximate_expectation(method::GenConstrainedUnscented, q::Tuple, f::F) where {F}
#     ms = mean.(q)
#     ds = size.(ms)
#     m = vcat(ms...)
#     V = vmapreduce(d -> typeof(d) <: Matrix ? sparse(d) : spdiagm([d]),blockdiag ,_cov.(q))
#     S = vcat(skewness.(q)...)
#     K = vcat(kurtosis.(q, false)...)
#     (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)
#     f_sigma = [f(__splitjoin(sp, ds)...) for sp in sigma_points] #
#     return sum(weights .* f_sigma)
# end

# function Unscented_approximate(method::GenConstrainedUnscented, g::G, means::Tuple, covs::Tuple,skewnesses::Tuple, kurtosises::Tuple) where {G}
#     (m, V, S, K, _) = unscented_statistics(method, Val(false), g, means, covs, skewnesses, kurtosises)
#     return (m, V, S, K)
# end

# function unscented_statistics(method::GenConstrainedUnscented, g::G, means::Tuple, covs::Tuple,skewnesses::Tuple, kurtosises::Tuple) where {G}
#     # By default we compute the `C` component, thus `Val(true)`
#     return unscented_statistics(method, Val(true), g, means, covs, skewnesses, kurtosises)
# end

# # Single univariate variable
# function unscented_statistics(method::GenConstrainedUnscented, ::Val{C}, g::G, means::Tuple{Real}, covs::Tuple{Real},skewnesses::Tuple{Real}, kurtosises::Tuple{Real}) where {C, G}
#     m = first(means)
#     V = first(covs)
#     S = first(skewnesses)
#     K = first(kurtosises)

#     (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)

#     g_sigma = g.(sigma_points)
#     m_tilde = sum(weights .* g_sigma)
#     V_tilde = sum(weights .* (g_sigma .- m_tilde) .^ 2)
#     S_tilde = sum(weights .* (g_sigma .- m_tilde) .^ 3)
#     K_tilde = sum(weights .* (g_sigma .- m_tilde) .^ 4)

#     # Compute `C_tilde` only if `C === true`
#     C_tilde = C ? sum(weights .* (sigma_points .- m) .* (g_sigma .- m_tilde)) : nothing

#     return (m_tilde, V_tilde, S_tilde, K_tilde, C_tilde)
# end

# # Single multivariate variable
# function unscented_statistics(method::GenConstrainedUnscented, ::Val{C}, g::G, means::Tuple{AbstractVector}, covs::Tuple{AbstractMatrix},skewnesses::Tuple{AbstractVector}, kurtosises::Tuple{AbstractVector} ) where {C, G}
#     m = first(means)
#     V = first(covs)
#     S = first(skewnesses)
#     K = first(kurtosises)

#     (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)

#     d = length(m)

#     g_sigma = g.(sigma_points)
#     @inbounds m_tilde = sum(weights[k + 1] * g_sigma[k + 1] for k in 0:(2d))
#     @inbounds V_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d))
#     @inbounds S_tilde = sum(weights[k + 1]*(g_sigma[k + 1] - m_tilde).^3 for k in 0:2d)
#     @inbounds K_tilde = sum(weights[k + 1]*(g_sigma[k + 1] - m_tilde).^4 for k in 0:2d)
#     # Compute `C_tilde` only if `C === true`
#     @inbounds C_tilde = C ? sum(weights[k + 1] * (sigma_points[k + 1] - m) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) : nothing

#     return (m_tilde, V_tilde, S_tilde, K_tilde, C_tilde)
# end

# # Multiple inbounds of possibly mixed variate type
# function unscented_statistics(method::GenConstrainedUnscented, ::Val{C}, g::G, ms::Tuple, Vs::Tuple,Ss::Tuple, Ks::Tuple) where {C, G}
#     ds     = size.(ms)
#     m      = vcat(ms...)
#     S      = vcat(Ss...)
#     K      = vcat(Ks...)
#     V      = vmapreduce(d -> typeof(d) <: Matrix ? sparse(d) : spdiagm([d]),blockdiag ,Vs)
#     (sigma_points, weights ) = sigma_points_weights(method, m, V, S, K)
#     g_sigma = [g(__splitjoin(sp, ds)...) for sp in sigma_points] # Unpack each sigma point in g

#     d = sum(prod.(ds)) # Dimensionality of joint
#     @inbounds m_tilde = sum(weights[k + 1] * g_sigma[k + 1] for k in 0:(2d)) # Vector
#     @inbounds V_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) # Matrix
#     @inbounds S_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde).^3 for k in 0:(2d))
#     @inbounds K_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde).^4 for k in 0:(2d))
#     # # Compute `C_tilde` only if `C === true`
#     @inbounds C_tilde = C ? sum(weights[k + 1] * (sigma_points[k + 1] - m) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) : nothing

#     return (m_tilde, V_tilde, S_tilde, K_tilde, C_tilde)
# end


# #Univariate sigma points and weights
# function sigma_points_weights(method::GenConstrainedUnscented, m::Real, V::Real,S::Real,K::Real)
#     lb = first(getlowerbound(method))
#     ub = first(getupperbound(method))
#     @assert lb < m < ub "Mean: $(m) should be within the upper bound: $(ub) and lower bound: $(lb)"
#     θ     = getθ(method)
#     l     = sqrt(V)
#     invL3 = inv(l^3) 
#     u     = (1/2)*(-S*invL3 + (1/V)*sqrt(4*K - 3*(S^2)/V))
#     v     = u + S*invL3
#     #initialize
#     constrained_sigma_points = (m, m - u*l, m + v*l)
#     vnew = v
#     ##lowerbound
   
#     if constrained_sigma_points[2] < lb
#         u = θ*abs((m - lb)/l)    
#     end
#     if constrained_sigma_points[3] < lb
#         vnew = θ*abs((lb - m)/l)
#     end
    

#     if vnew == v
#         vnew = u + S*inv(l^3) 
#         constrained_sigma_points = (m, m - u*l, m + vnew*l)
#     else
#         constrained_sigma_points = (m, m - u*l, m + vnew*l)
#     end
#     vnew2 = vnew
#     ##upperbound
    
#     if constrained_sigma_points[2] > ub
#         u = θ*abs((m - ub)/l) 
#     end
#     if constrained_sigma_points[3] > ub
#         vnew2 = θ*abs((ub - m)/l)
#     end
    
#     if vnew2 == vnew
#         vnew2                    = u + S*inv(l^3) 
#         constrained_sigma_points = (m, m - u*l, m + vnew2*l)
#         auxiliary                = inv(vnew2*(u+vnew2))
#         constrained_weights      = (1-auxiliary*(vnew2/u +1)  , (vnew2/u)*auxiliary, auxiliary)
#     else
#         constrained_sigma_points = (m, m - u*l, m + vnew2*l)
#         auxiliary                = inv(vnew2*(u+vnew))
#         constrained_weights      = (1-auxiliary*(vnew2/u +1)  , (vnew2/u)*auxiliary, auxiliary)
#     end
#     return constrained_sigma_points, constrained_weights

# end

# #Multivariate sigma points and weights
# function sigma_points_weights(method::GenConstrainedUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector,K::AbstractVector)
#     lb = vcat(getlowerbound(method)...)
#     ub = vcat(getupperbound(method)...)
#     @assert all(lb .< m .< ub) "Mean: $(m) should be within the upper bound: $(ub) and lower bound: $(lb)"
#     dim = length(m)
#     T = promote_type(eltype(m), eltype(V),eltype(S),eltype(K))
#     θ = getθ(method)
#     L = cholsqrt(V)
#     L3    = L.^3
#     invL3 = cholinv(L3)
#     invL4 = cholinv(L3 .* L3)
#     determinant = 4*invL4*K - 3*(invL3*S).^2
#     u           = (1/2)*(-invL3*S + sqrt.(determinant ) )
#     v           = u + invL3*S
#     constrained_sigma_points = Vector{Vector{T}}(undef, 2 * dim + 1)
#     weights      = Vector{T}(undef, 2 * dim+ 1)
#     constrained_sigma_points[1] = m
#     @inbounds for i in 1:dim
#         @views constrained_sigma_points[i+1]  = m - L[:, i]*u[i]
#         @views constrained_sigma_points[i+dim+1] = m + L[:, i]*v[i]
#     end
#     vnew = v
#     ## lower part
#     @inbounds for i=1:dim
#         if any(constrained_sigma_points[i+1] .< lb)
#             @views u[i] = θ*minimum(abs.((m - lb) ./ L[:,i]))
#         end
#         if any(constrained_sigma_points[dim+i+1] < lb)
#             @views vnew[i] = θ*minimum(abs.((lb - m) ./ L[:,i]))
#         end
#     end
    
#     if vnew == v
#         vnew = u + invL3*S
#         @inbounds for i in 1:dim
#             @views constrained_sigma_points[i+1]  = m - L[:, i]*u[i]
#             @views constrained_sigma_points[i+dim+1] = m + L[:, i]*vnew[i]
#         end
#     else 
#         @inbounds for i in 1:dim
#             @views constrained_sigma_points[i+1]     = m - L[:, i]*u[i]
#             @views constrained_sigma_points[i+dim+1] = m + L[:, i]*vnew[i]
#         end
#     end 
#     vnew2 = vnew 
#     ## upper bound
#     @inbounds for i=1:dim
#         if any( constrained_sigma_points[i+1] .> ub )
#             @views u[i] = θ*minimum(abs.((m - ub) ./ L[:,i]))
#         end
#         if any( constrained_sigma_points[dim+i+1] .> ub)
#             @views vnew2[i] = θ*minimum(abs.((ub - m) ./ L[:,i]))
#         end
#     end
#     if vnew2 == vnew
#         vnew2 = u + invL3*S
#         @inbounds for i in 1:dim
#             @views constrained_sigma_points[i+1]  = m - L[:, i]*u[i]
#             @views constrained_sigma_points[i+dim+1] = m + L[:, i]*vnew2[i]
#         end
#         @inbounds weights[dim+2:end] .=  1 ./ vnew2 ./ ( u + vnew2) 
#         @inbounds weights[2:dim+1]   .= weights[dim+2:end] .* (vnew2 ./ u)
#         weights[1] = 1 - sum(weights[2:end])
#     else
#         @inbounds for i in 1:dim
#             @views constrained_sigma_points[i+1]  = m - L[:, i]*u[i]
#             @views constrained_sigma_points[i+dim+1] = m + L[:, i]*vnew2[i]
#         end
#         @inbounds weights[dim+2:end] .=  1 ./ vnew2 ./ ( u + vnew2) 
#         @inbounds weights[2:dim+1]   .= weights[dim+2:end] .* (vnew2 ./ u)
#         weights[1] = 1 - sum(weights[2:end])
#     end

#     return (constrained_sigma_points, weights)
# end

