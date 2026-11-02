@testitem "every protocol name is public and none is exported" begin
    const PROTOCOL = [
        :expectation, :prepare_expectation, :Prepared, :NoBackendError,
        :supports, :which_backend,
        :ExpectationLabel, :Exact, :Stochastic, :haslabel, :isexact, :isstochastic, :combine_label,
        :withrng, :getrng,
        :Logpdf,
        :PointSet, :ProductPointSet,
        :FirstOf,
        :DynamicRouter, :BackendRegistry, :register_backend!,
    ]

    for name in PROTOCOL
        @test isdefined(ProtoBase, name)
        if VERSION >= v"1.11"
            @test Base.ispublic(ProtoBase, name)
            @test !Base.isexported(ProtoBase, name)
        else
            @test name ∉ names(ProtoBase) # before 1.11 `names` lists exported names only
        end
    end
end

@testitem "loading next to RxInfer's names does not clash" begin
    # MessagePassingRulesBase and ReactiveMP export `Stochastic` and `isstochastic` for node
    # types; ProtoBase exports nothing, so a module may use both
    module UsesBoth
        using ProtoBase
        struct Stochastic end # stands in for the node type exported by ReactiveMP
        isstochastic(::Stochastic) = true
        const label = ProtoBase.Stochastic()
    end
    @test UsesBoth.isstochastic(UsesBoth.Stochastic())
    @test UsesBoth.label isa ProtoBase.ExpectationLabel
end
