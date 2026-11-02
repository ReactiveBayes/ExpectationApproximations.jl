@testitem "loading next to RxInfer: nothing clashes" begin
    module WithRxInfer
        using RxInfer, ProtoBase, ProtoBackends
    end
    using RxInfer, ProtoBase

    # the node types keep their names, the labels are reached qualified
    @test Core.eval(WithRxInfer, :Stochastic) === RxInfer.Stochastic
    @test Core.eval(WithRxInfer, :isstochastic) === RxInfer.isstochastic
    @test ProtoBase.Stochastic !== RxInfer.Stochastic
    @test !isdefined(WithRxInfer, :expectation) || Core.eval(WithRxInfer, :expectation) !== ProtoBase.expectation
    @test !isdefined(WithRxInfer, :GaussHermite)
end
