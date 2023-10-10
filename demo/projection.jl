### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 5f65c9c4-52d3-11ee-179d-fdc06aeeaf0e
begin
	using Pkg
	using Revise
	Pkg.activate("/Users/mykola/repos/biaslab/ExpectationApproximations.jl")
	using ExpectationApproximations
	using ExponentialFamily
	using Distributions
	using Random
	using StableRNGs
	using Optimisers
end

# ╔═╡ 8d948d11-8c5f-4378-9b75-336828fa2f6e
begin
	cvmp_method = CVMP(
		StableRNG(42), 
		100,
		Optimisers.Adam(0.01),
		ForwardDiffGrad(),
		10000,
		Val(true),
		true,
		SimpleInitStrategy()
	)
end

# ╔═╡ 897edd49-c5a0-4363-b00d-e8356ac0e55f
begin
	ef = ExponentialFamily.ExponentialFamilyDistribution(LogNormal, [0, 4])
	positive_measure = ExponentialFamily.ExponentialFamilyDistribution(NormalMeanVariance, [1.0, 4])
	marginal = ExponentialFamily.ExponentialFamilyDistribution(NormalMeanVariance, [1, 8])
end

# ╔═╡ 03d91c9e-0c5c-4caa-8d93-3e95a815f51d
convert(
	Distribution, 
	ExponentialFamily.ExponentialFamilyDistribution(NormalMeanVariance, [1, 8])
)

# ╔═╡ 6fe8f65d-fc39-43b0-8f42-5cf066304536
cvmp_factor = CVMPFactor(ef, positive_measure, marginal)

# ╔═╡ f6156da7-6c3d-41fe-aa15-875ebab7b077
product = ExpectationApproximations.approximate_distribution(
	cvmp_method,
	cvmp_factor,
	exp
)

# ╔═╡ Cell order:
# ╠═5f65c9c4-52d3-11ee-179d-fdc06aeeaf0e
# ╠═8d948d11-8c5f-4378-9b75-336828fa2f6e
# ╠═897edd49-c5a0-4363-b00d-e8356ac0e55f
# ╠═03d91c9e-0c5c-4caa-8d93-3e95a815f51d
# ╠═6fe8f65d-fc39-43b0-8f42-5cf066304536
# ╠═f6156da7-6c3d-41fe-aa15-875ebab7b077
