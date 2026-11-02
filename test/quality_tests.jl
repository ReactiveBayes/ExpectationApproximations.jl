@testitem "quality:aqua" tags = [:quality] begin
    using Aqua
    # As before the move to TestItemRunner. The package has no compat entries yet, which Aqua
    # reports; the rewrite of ExpectationApproximations (U2 of PLAN.md) adds them, together with
    # JET and the doctests of the template.
    Aqua.test_all(ExpectationApproximations; ambiguities = false, piracies = false)
end
