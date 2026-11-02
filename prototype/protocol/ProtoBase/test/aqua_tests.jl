@testitem "Aqua, with ambiguity and piracy checks" begin
    using Aqua, ProtoBase

    Aqua.test_all(ProtoBase; ambiguities = true, piracies = true)
end
