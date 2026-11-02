function weighted_sum(::Type{FromTo{T, T}}, method, points, weights, transformedpoints, f::F, ds) where {F, T <: Univariate}
    isnothing(ds) ? map!((x, y) -> y * f(x), transformedpoints, points, weights) : map!((x, y) -> y * f(__splitjoin(x, ds)...), transformedpoints, points, weights)
    return sum(transformedpoints)
end

function weighted_sum(::Type{FromTo{Tin, Tout}}, method, points, weights, transformedpoints, f::F, ds) where {F, Tout <: Univariate, Tin <: Multivariate}
    if isnothing(ds)
        map!((x, y) -> y * f(x), transformedpoints, eachcol(points), weights)
    else
        map!((x, y) -> y * f(__splitjoin(x, ds)...), transformedpoints, eachcol(points), weights)
    end
    return sum(transformedpoints)
end

function weighted_sum(::Type{FromTo{Tin, Tout}}, method, points, weights, transformedpoints, f::F, ds) where {F, Tout <: Multivariate, Tin <: Univariate}
    if isnothing(ds)
        map!((x, y) -> lmul!(y, f(x)), eachcol(transformedpoints), points, weights)
    else
        map!((x, y) -> lmul!(y, f(__splitjoin(x, ds)...)), eachcol(transformedpoints), points, weights)
    end
    result = zeros(first(method.outputdim))
    for elem in 1:first(method.outputdim)
        result[elem] = sum(@views transformedpoints[elem, :])
    end

    return result
end

function weighted_sum(::Type{FromTo{T, T}}, method, points, weights, transformedpoints, f::F, ds) where {F, T <: Multivariate}
    if isnothing(ds)
        map!((x, y) -> lmul!(y, f(x)), eachcol(transformedpoints), eachcol(points), weights)
    else
        map!((x, y) -> lmul!(y, f(__splitjoin(x, ds)...)), eachcol(transformedpoints), eachcol(points), weights)
    end
    result = zeros(first(method.outputdim))
    for elem in 1:first(method.outputdim)
        result[elem] = sum(@views transformedpoints[elem, :])
    end

    return result
end

function weighted_sum(::Type{FromTo{Tin, Tout}}, method, points, weights, transformedpoints, f::F, ds) where {F, Tout <: Matrixvariate, Tin <: Univariate}
    if isnothing(ds)
        map!((x, y) -> lmul!(y, f(x)), eachslice(transformedpoints, dims = 3), points, weights)
    else
        map!((x, y) -> lmul!(y, f(__splitjoin(x, ds)...)), eachslice(transformedpoints, dims = 3), points, weights)
    end

    result = zeros(first(method.outputdim), last(method.outputdim))

    for i in 1:first(method.outputdim)
        for j in 1:last(method.outputdim)
            result[i, j] = sum(@views transformedpoints[i, j, :])
        end
    end

    return result
end

function weighted_sum(::Type{FromTo{Tin, Tout}}, method, points, weights, transformedpoints, f::F, ds) where {F, Tout <: Matrixvariate, Tin <: Multivariate}
    if isnothing(ds)
        map!((x, y) -> lmul!(y, f(x)), eachslice(transformedpoints, dims = 3), eachcol(points), weights)
    else
        map!((x, y) -> lmul!(y, f(__splitjoin(x, ds)...)), eachslice(transformedpoints, dims = 3), eachcol(points), weights)
    end

    result = zeros(first(method.outputdim), last(method.outputdim))
    for i in 1:first(method.outputdim)
        for j in 1:last(method.outputdim)
            result[i, j] = sum(@views transformedpoints[i, j, :])
        end
    end

    return result
end
