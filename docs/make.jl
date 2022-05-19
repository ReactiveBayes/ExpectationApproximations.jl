using ExpectationApproximations
using Documenter

DocMeta.setdocmeta!(ExpectationApproximations, :DocTestSetup, :(using ExpectationApproximations); recursive=true)

makedocs(;
    modules=[ExpectationApproximations],
    authors="Bagaev Dmitry <bvdmitri@gmail.com> and contributors",
    repo="https://github.com/bvdmitri/ExpectationApproximations.jl/blob/{commit}{path}#{line}",
    sitename="ExpectationApproximations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bvdmitri.github.io/ExpectationApproximations.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bvdmitri/ExpectationApproximations.jl",
    devbranch="main",
)
