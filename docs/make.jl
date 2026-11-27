# The documentation site of ExpectationApproximations. Build it with `make docs`, or
# `julia --project=docs docs/make.jl`.
using Documenter
using ExpectationApproximations

# Plots draws without a display on CI and in `make docs`
ENV["GKSwstype"] = "100"

# The docstrings' examples use the names as a user would: Distributions' families, BayesBase's
# protocol, and this package as `EA`
DocMeta.setdocmeta!(
    ExpectationApproximations,
    :DocTestSetup,
    :(using Distributions; import BayesBase; import ExpectationApproximations as EA);
    recursive = true,
)

makedocs(
    modules = [ExpectationApproximations],
    # Every docstring appears on a page.
    checkdocs = :all,
    authors = "Bagaev Dmitry <bvdmitri@gmail.com> and contributors",
    sitename = "ExpectationApproximations.jl",
    pages = [
        "Home" => "index.md",
        "Computing expectations" => "tutorial.md",
        "How a backend is chosen" => "routing.md",
        "Choosing a backend" => "backends.md",
        "Gaussian approximations of a density" => "projections.md",
        "Moments through functions" => "moments.md",
        "Constrained variables" => "transforms.md",
        "Quadrature rules" => "rules.md",
        "Writing a backend" => "writing.md",
        "Migrating" => "migrating.md",
    ],
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
