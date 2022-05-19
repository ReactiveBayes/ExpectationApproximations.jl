using Revise, Jive
using Test
using ExpectationApproximations

trigger = function (path)
    printstyled("changed ", color = :cyan)
    println(path)
    revise()
    runtests(@__DIR__, skip = ["watchtests.jl"])
end

watch(trigger, @__DIR__, sources = [pathof(ExpectationApproximations)])

trigger("")

Base.JLOptions().isinteractive == 0 && wait()