@testitem "loading next to RxInfer: nothing clashes" tags = [:collisions] begin
    module WithRxInfer
    using RxInfer, BayesBase, ProtoBackends
    end
    using RxInfer, BayesBase

    # the node types keep their names, the labels are reached qualified
    @test Core.eval(WithRxInfer, :Stochastic) === RxInfer.Stochastic
    @test Core.eval(WithRxInfer, :isstochastic) === RxInfer.isstochastic
    @test BayesBase.Stochastic !== RxInfer.Stochastic
    @test !isdefined(WithRxInfer, :expectation) || Core.eval(WithRxInfer, :expectation) !== BayesBase.expectation
    @test !isdefined(WithRxInfer, :GaussHermite)
end
