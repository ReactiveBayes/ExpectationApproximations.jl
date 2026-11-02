@testitem "Aqua, with ambiguity and piracy checks" begin
    using Aqua, ProtoBackends

    Aqua.test_all(ProtoBackends; ambiguities = true, piracies = true)
end

@testitem "every backend name is public and none is exported" begin
    using ProtoBackends

    for name in (:ClosedForms, :GaussHermite, :QuantileLegendre, :MonteCarlo, :default_router)
        @test isdefined(ProtoBackends, name)
        @test Base.ispublic(ProtoBackends, name)
        @test !Base.isexported(ProtoBackends, name)
    end
end
