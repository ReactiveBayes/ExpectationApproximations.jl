# The documentation site of ExpectationApproximations. Build it with `make docs`, or
# `julia --project=docs docs/make.jl`.
using Documenter
using ExpectationApproximations

DocMeta.setdocmeta!(
    ExpectationApproximations, :DocTestSetup, :(using ExpectationApproximations); recursive = true
)

makedocs(
    modules = [ExpectationApproximations],
    # Every docstring appears on a page.
    checkdocs = :all,
    authors = "Bagaev Dmitry <bvdmitri@gmail.com> and contributors",
    sitename = "ExpectationApproximations.jl",
    pages = ["Home" => "index.md", "Backends" => "backends.md", "Projections" => "projections.md", "Domain transforms" => "transforms.md", "Quadrature rules" => "rules.md", "Conformance suite" => "testing.md"],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://reactivebayes.github.io/ExpectationApproximations.jl",
        edit_link = "main",
    ),
)

deploydocs(
    repo = "github.com/ReactiveBayes/ExpectationApproximations.jl.git",
    devbranch = "main",
    # A preview for every pull request but CompatHelper's, whose documentation nobody reviews;
    # `DocPreviewCleanup.yml` deletes it when the pull request closes.
    push_preview = !startswith(get(ENV, "GITHUB_HEAD_REF", ""), "compathelper/"),
)
