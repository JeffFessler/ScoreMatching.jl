#=
# [Score Matching overview](@id 01-overview)

This page introduces the Julia package
[`ScoreMatching`](https://github.com/JeffFessler/ScoreMatching.jl).

This page was generated from a single Julia file:
[01-overview.jl](@__REPO_ROOT_URL__/01-overview.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`01-overview.ipynb`](@__NBVIEWER_ROOT_URL__/01-overview.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`01-overview.ipynb`](@__BINDER_ROOT_URL__/01-overview.ipynb).


# ### Setup

# Packages needed here.

using ScoreMatching
using MIRTjim: jim, prompt
using Distributions: Distribution, Normal, MixtureModel, logpdf, pdf
using Distributions: Cauchy, Gamma, Logistic, TDist
import Distributions: logpdf, pdf
import ForwardDiff
using LinearAlgebra: tr, norm
using LaTeXStrings
using Random: seed!; seed!(0)
using StatsBase: mean
using Optim: optimize, BFGS, Fminbox
import Optim: minimizer
#src import ReverseDiff
using Plots: plot, plot!, scatter, default, gui
using InteractiveUtils: versioninfo
default(label="", markerstrokecolor=:auto)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:pause);


#=
## Overview

Given ``T``
[IID](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
training data samples
``\mathbf{x}_1, …, \mathbf{x}_T ∈ \mathbb{R}^N``,
we often want to find the parameters
``\mathbf{θ}``
of a model distribution
``p(\mathbf{x}; \mathbf{θ})``
that "best fit" the data.

Maximum-likelihood estimation
is impractical for complicated models
where the normalizing constant is intractable.

[Hyvärinen 2005](http://jmlr.org/papers/v6/hyvarinen05a.html)
proposed an alternative called
_score matching_
that circumvents
needing to find the normalizing constant.

The idea behind the score matching approach
to model fitting is
```math
\hat{\mathbf{θ}} = \arg \min_{\mathbf{θ}}
J(\mathbf{θ})
,\qquad
J(\mathbf{θ}) =
\frac{1}{T} ∑_{t=1}^T
\| \mathbf{s}(\mathbf{x}_t; \mathbf{θ}) - \mathbf{s}(\mathbf{x}_t) \|^2
```
where
``
\mathbf{s}(\mathbf{x}; \mathbf{θ}) =
\nabla_{\mathbf{x}} \log p(\mathbf{x}; \mathbf{θ})
``
is the _score function_
of the model distribution,
and
``
\mathbf{s}(\mathbf{x}) =
\nabla_{\mathbf{x}} \log p(\mathbf{x})
``
is the _score function_
of the (typically unknown) data distribution.

[Vincent, 2011](https://doi.org/10.1162/NECO_a_00142)
calls this approach
_explicit score matching_ (ESM).

## Illustration

For didactic purposes,
we illustrate explicit score matching
by fitting samples from a
[Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
to a mixture of gaussians.

=#

# Some convenience methods
logpdf(d::Distribution) = x -> logpdf(d, x)
pdf(d::Distribution) = x -> pdf(d, x)
derivative(f::Function) = x -> ForwardDiff.derivative(f, x)
gradient(f::Function) = x -> ForwardDiff.gradient(f, x)
## hessian(f::Function) = x -> ForwardDiff.hessian(f, x)
score(d::Distribution) = derivative(logpdf(d))
score_deriv(d::Distribution) = derivative(score(d)) # scalar x only


# Generate training data
T = 100
data_disn = :(Gamma(8, 1))
data_dis = eval(data_disn)
data_score = derivative(logpdf(data_dis))
data = rand(data_dis, T)
xlims = (-1, 25)
xticks = [0, 8, 24]
pf = scatter(data, zeros(T); xlims, xticks, color=:black)


#=
To perform unconstrained minimization
of a ``D``-component mixture,
the following mapping from ``\mathbb{R}^{D-1}``
to the ``D``-dimensional simplex is helpful.
It is the inverse of the
[additive logratio transform](https://en.wikipedia.org/wiki/Compositional_data#Additive_logratio_transform).
(It is related to the
[softmax function](https://en.wikipedia.org/wiki/Softmax_function).
=#

function map_r_s(y::AbstractVector; scale::Real = 1.0)
    y = scale * [y; 0]
    y .-= maximum(y) # for numerical stability
    p = exp.(y)
    return p / sum(p)
end
map_r_s(y::Real...) = map_r_s([y...])

y1 = range(-1,1,101) * 9
y2 = range(-1,1,101) * 9
tmp = map_r_s.(y1, y2')
pj = jim(y1, y2, tmp; title="Simplex parameterization", nrow=1)


# Define model distribution

nmix = 3 # how many gaussians in the mixture model
function model(θ ;
    σmin::Real = 1,
    σmax::Real = 19,
)
    mu = θ[1:nmix]
    sig = θ[nmix .+ (1:nmix)]
    any(<(σmin), sig) && throw("too small σ")
    any(>(σmax), sig) && throw("too big σ $sig")
    ## sig = σmin .+ exp.(sig) # ensure σ > 0
    ## sig = @. σmin + (σmax - σmin) * (tanh(sig/2) + 1) / 2 # "constraints"
    p = map_r_s(θ[2nmix .+ (1:(nmix-1))])
    tmp = [(μ,σ) for (μ,σ) in zip(mu, sig)]
    mix = MixtureModel(Normal, tmp, p)
    return mix
end;


# Define explicit score-matching cost function
function cost_esm2(x::AbstractVector{<:Real}, θ)
    model_score = score(model(θ))
    return (0.5/T) * sum(abs2, model_score.(x) - data_score.(x))
end;

# Minimize this explicit score-matching cost function:
β = 0e-4 # optional small regularizer to ensure coercive
cost_esm1 = (θ) -> cost_esm2(data, θ) + β * 0.5 * norm(θ)^2;

# Initial crude guess of mixture model parameters
θ0 = Float64[5, 7, 9, 1.5, 1.5, 1.5, 0, 0]; # Gamma
#src θ0 = Float64[mean(data) .+ [-2, 0, 2]; -2; -2; -2; 0; 0];
#src θ0 = Float64[10, 15, 2, 2, 0];
#src θ0 = Float64[10, 15, -2, -2, 0];

# Plot data pdf and initial model pdf
plot!(pf, pdf(data_dis); xlims, xticks, label="$data_disn pdf",
    color = :black,
    xlabel = L"x",
    ylabel = L"p(x) \ \mathrm{ and } \ p(x;θ)",
)
plot!(pf, pdf(model(θ0)), label = "Initial Gaussian mixture", color=:blue)

#
prompt()

# Check descent and non-convexity
if false
    tmp = gradient(cost_esm1)(θ0)
    a = range(0, 9, 101)
    h = a -> cost_esm1(θ0 - a * tmp)
    plot(a, log.(h.(a)))
end


# ## Explicit score matching (impractical)

lower = [fill(0, nmix); fill(1.0, nmix); fill(-Inf, nmix-1)]
upper = [fill(Inf, nmix); fill(Inf, nmix); fill(Inf, nmix-1)]
opt_esm = optimize(cost_esm1, lower, upper, θ0, Fminbox(BFGS());
 autodiff = :forward)
##opt_esm = optimize(cost_esm1, θ0, BFGS(); autodiff = :forward) # unconstrained
θesm = minimizer(opt_esm)

plot!(pf, pdf(model(θesm)), label = "ESM Gaussian mixture", color=:green)

#
prompt()


#=
Plot the data score and model score functions
to see how well they match.
The largest mismatch is in the tails of the distribution
where there are few (if any) data points.
=#
ps = plot(data_score; xlims=(1,20), label = "Data score function",
    xticks=[1,20], xlabel=L"x", color=:black)
plot!(ps, score(model(θesm)); label = "ESM score function", color=:green)


#
prompt()


#=
## Maximum-likelihood estimation

This toy example is simple enough
that we can apply ML estimation to it directly.
In fact, ML estimation is a seemingly more practical optimization problem
than score matching in this case.

As expected,
ML estimation leads to a lower negative log-likelihood.
=#


negloglike(θ) = (-1/T) * sum(logpdf(model(θ)), data)
opt_ml = optimize(negloglike, lower, upper, θ0, Fminbox(BFGS()); autodiff = :forward)
##opt_ml = optimize(negloglike, θ0, BFGS(); autodiff = :forward)
θml = minimizer(opt_ml)
negloglike.([θml, θesm, θ0])

#=
Curiously,
ML estimation here leads to much worse fits to the pdf
than score matching,
even though we initialized the ML optimizer
with the score-matching parameters.
Perhaps the landscape of the log-likelihood
is less well-behaved
than that of the SM cost.
=#
plot!(pf, pdf(model(θml)), label = "ML Gaussian mixture", color=:magenta)
plot!(ps, score(model(θml)), label = "ML score function", color=:magenta)
plot(pf, ps)

#
prompt()


#=
## Implicit score matching (more practical)

The above ESM fitting process
used `score(data_dis)`,
the score-function of the data distribution,
which is unknown in practical situations.

[Hyvärinen 2005](http://jmlr.org/papers/v6/hyvarinen05a.html)
derived the following more practical cost function
that is independent of the unknown data score function:
```math
J_{\mathrm{ISM}}(\mathbf{θ}) =
\frac{1}{T} ∑_{t=1}^T
∑_{i=1}^N ∂_i s_i(\mathbf{x}_t; \mathbf{θ})
 + \frac{1}{2} | s_i(\mathbf{x}_t; \mathbf{θ}) |^2,
```
ignoring a constant that is independent of ``θ,``
where
```math
∂_i s_i(\mathbf{x}; \mathbf{θ})
=
\frac{∂}{∂ x_i} s_i(\mathbf{x}; \mathbf{θ})
=
\frac{∂^2}{∂ x_i^2} \log p(\mathbf{x}; \mathbf{θ}).
```

(For large models
this version is still a bit impractical
because it depends on the diagonal
elements of the Hessian
of the log prior.
Subsequent pages deal with that issue.)

[Vincent, 2011](https://doi.org/10.1162/NECO_a_00142)
calls this approach
_implicit score matching_ (ISM).
=#


# Implicit score-matching cost function
function cost_ism2(x::AbstractVector{<:Real}, θ)
    tmp = model(θ)
    model_score = score(tmp)
    return (1/T) * (sum(score_deriv(tmp), x) +
        0.5 * sum(abs2 ∘ model_score, x))
end;

# Minimize this implicit score-matching cost function:
cost_ism1 = (θ) -> cost_ism2(data, θ)
opt_ism = optimize(cost_ism1, lower, upper, θ0, Fminbox(BFGS()); autodiff = :forward)
##opt_ism = optimize(cost_ism1, θ0, BFGS(); autodiff = :forward)
θism = minimizer(opt_ism)
cost_ism1.([θism, θesm, θml])

#
plot!(pf, pdf(model(θism)), label = "ISM Gaussian mixture", color=:cyan)
plot!(ps, score(model(θism)), label = "ISM score function", color=:cyan)
pfs = plot(pf, ps)

#
prompt()


#=
Curiously the supposedly equivalent ISM cost function works much worse.
Like the ML estimate,
the first two ``σ`` values are stuck at the `lower` limit.
Could it be local extrema?
More investigation is needed!

Ideally,
the ESM and ISM cost functions
should differ by a constant independent of ``θ``.
=#

tmp = [θ0, θesm, θml, θism]
cost_esm1.(tmp) - cost_ism1.(tmp)

# todo - bug since non-constant?

#=
[Kingma & LeCun, 2010](https://doi.org/10.5555/2997189.2997315)
reported some instability of ISM
and suggested a regularized version
corresponding to the following cost function:
xx
```math
J_{\mathrm{RSM}}(\mathbf{θ}) =
J_{\mathrm{ISM}}(\mathbf{θ}) + λ R(\mathbf{θ})
,\quad
R(\mathbf{θ}) =
\frac{1}{T} ∑_{t=1}^T
∑_{i=1}^N | ∂_i s_i(\mathbf{x}_t; \mathbf{θ}) |^2.
```
=#


# Regularized ISM cost function
function cost_rsm2(x::AbstractVector{<:Real}, θ, λ)
    mod = model(θ)
    model_score = score(mod)
    tmp = score_deriv(mod).(x)
    R = sum(abs2, tmp)
    J_ism = sum(tmp) + 0.5 * sum(abs2 ∘ model_score, x)
    return (1/T) * (J_ism + λ * R)
end;

# Minimize this regularized ISM cost function:
λ = 2e0
cost_rsm1 = (θ) -> cost_rsm2(data, θ, λ)

#upper_rsm = [fill(Inf, nmix); fill(19, nmix); fill(Inf, nmix-1)]
upper_rsm = upper
opt_rsm = optimize(cost_rsm1, lower, upper_rsm, θ0, Fminbox(BFGS());
 autodiff = :forward)
θrsm = minimizer(opt_rsm)
cost_rsm1.([θrsm, θ0, θism, θesm, θml])

#
plot!(pf, pdf(model(θism)), label = "RSM Gaussian mixture", color=:red)
plot!(ps, score(model(θism)), label = "RSM score function", color=:red)
pfs = plot(pf, ps)

#
prompt()

#=
Sadly the RSM approach did not help much here.
Increasing ``λ`` led to `optimize` errors.
=#


# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
