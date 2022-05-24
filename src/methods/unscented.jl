
struct Unscented{R}
    rule :: R
end

struct SupportsLogApproximation end
struct LogApproximationSupportUndefined end

rule_log_approximation_support(rule) = LogApproximationSupportUndefined()