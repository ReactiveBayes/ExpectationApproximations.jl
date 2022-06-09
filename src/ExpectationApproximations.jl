module ExpectationApproximations

    include("transforms.jl")
    include("log.jl")
    include("approximate.jl")

    include("kernels/raw_moment.jl")
    include("kernels/central_moment.jl")

    include("methods/unscented/unscented.jl")

end
