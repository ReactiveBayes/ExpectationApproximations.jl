using ExpectationApproximations
using Documenter

DocMeta.setdocmeta!(ExpectationApproximations, :DocTestSetup, :(using ExpectationApproximations); recursive=true)

makedocs(;
    modules  = [ ExpectationApproximations ],
    strict   = [ :doctest, :eval_block, :example_block, :meta_block, :parse_error, :setup_block ],
    authors  = "Bagaev Dmitry <bvdmitri@gmail.com> and contributors",
    repo     = "https://github.com/reactivebayes/ExpectationApproximations.jl/blob/{commit}{path}#{line}",
    sitename = "ExpectationApproximations.jl",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://reactivebayes.github.io/ExpectationApproximations.jl",
        assets     = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API"  => [
            "Common" => "common.md"
        ]
    ],
)

deploydocs(;
    repo      = "github.com/reactivebayes/ExpectationApproximations.jl",
    devbranch = "main",
)
