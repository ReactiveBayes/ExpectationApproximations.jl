

function approximate(method, q, f)
    error("`f` should be either `Pdf` or `LogPdf`")
end

function approximate(method, q, f::Union{Pdf, LogPdf})
    # TODO add checks for domain
    return call_approximate(method, q, f)
end