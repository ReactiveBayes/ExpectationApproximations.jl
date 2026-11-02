@testitem "quality:aqua" tags = [:quality] begin
    using Aqua
    Aqua.test_all(ExpectationApproximations; deps_compat = (; check_extras = true, check_weakdeps = true))
end

@testitem "quality:jet" tags = [:quality] begin
    # JET follows Julia's compiler closely, and its releases for Julia 1.10 and 1.11 cannot analyse
    # every package; it runs from 1.12 on.
    if VERSION >= v"1.12"
        using JET
        JET.test_package(ExpectationApproximations; target_modules = (ExpectationApproximations,))
    end
end

@testitem "quality:doctests" tags = [:quality] begin
    using Documenter
    DocMeta.setdocmeta!(
        ExpectationApproximations, :DocTestSetup, :(using ExpectationApproximations); recursive = true
    )
    doctest(ExpectationApproximations; manual = false)
end

@testitem "quality:nothing is exported" tags = [:quality] begin
    # Every name is public and used qualified, as in BayesBase's expectation protocol
    if VERSION >= v"1.11"
        exported = filter(name -> Base.isexported(ExpectationApproximations, name), names(ExpectationApproximations))
        @test exported == [:ExpectationApproximations]
    else
        @test names(ExpectationApproximations) == [:ExpectationApproximations] # exported names only
    end
end
