function init_model(x::AbstractArray)
    return (params = (x,),)
end

hybridlinearize(vector::AbstractVector) = vector
hybridlinearize(matrix::AbstractMatrix) = eachcol(matrix)