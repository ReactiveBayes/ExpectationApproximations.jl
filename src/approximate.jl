export approximate

function approximate(method, q, f)
    return call_approximate(method, q, f)
end

function mean_var(method, q, f)
    return call_mean_var(method, q, f)
end