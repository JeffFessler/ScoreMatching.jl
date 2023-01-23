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
using Distributions: Distribution, Gamma, Normal, MixtureModel, logpdf, pdf
import Distributions: logpdf, pdf
import ForwardDiff
using LinearAlgebra: tr
using LaTeXStrings
using Random: seed!; seed!(0)
using Optim: optimize, BFGS, Fminbox
import Optim: minimizer
#src import ReverseDiff
using Plots: plot, plot!, scatter, default, gui
using InteractiveUtils: versioninfo
default(label="", markerstrokecolor=:auto)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


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


## Illustration

For didactic purposes,
we illustrate this basic version of score matching
by fitting samples from a
[Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
to a mixture of gaussians.

=#

# Some convenience methods
logpdf(d::Distribution) = x -> logpdf(d, x)
pdf(d::Distribution) = x -> pdf(d, x)
derivative(f::Function) = x -> ForwardDiff.derivative(f, x)
gradient(f::Function) = x -> ForwardDiff.gradient(f, x)
#src hessian(f::Function) = x -> ForwardDiff.hessian(f, x)
score(d::Distribution) = derivative(logpdf(d))
score_deriv(d::Distribution) = derivative(score(d)) # scalar x only

# Generate training data
T = 100
data_dis = Gamma(8, 1.0)
data_score = derivative(logpdf(data_dis))
data = rand(data_dis, T)
pd = scatter(data, zeros(T))


#=
To perform unconstrained minimization
of a ``D``-component mixture,
the following mapping from ``\mathbb{R}^{D-1}``
to the ``D``-dimensional simplex is helpful.
It is the inverse of the
[additive logratio transform](https://en.wikipedia.org/wiki/Compositional_data#Additive_logratio_transform).
=#

function map_r_s(y::AbstractVector)
    p = exp.([y; 0])
    return p / sum(p)
end
map_r_s(y::Real...) = map_r_s([y...])

y1 = range(-1,1,101) * 9
y2 = range(-1,1,101) * 9
tmp = map_r_s.(y1, y2')
jim(y1, y2, tmp; title="Simplex parameterization", nrow=1)


# Define model distribution

nmix = 3 # how many gaussians in the mixture model
function model(θ ; σmin::Real=1e-2)
    mu = θ[1:nmix]
    sig = θ[nmix .+ (1:nmix)]
    any(<(σmin), sig) && throw("bad σ")
#   sig = σmin .+ exp.(sig) # ensure σ > 0
    p = map_r_s(θ[2nmix .+ (1:(nmix-1))])
    tmp = [(μ,σ) for (μ,σ) in zip(mu, sig)]
    return MixtureModel(Normal, tmp, p)
end;

# Define score-matching cost function
function cost_sm2(x::AbstractVector{<:Real}, θ)
    model_score = score(model(θ))
    return sum(abs2, model_score.(x) - data_score.(x)) / T
end;

# Minimize this score-matching cost function:
cost_sm1 = (θ) -> cost_sm2(data, θ);

# Initial crude guess of mixture model parameters
θ0 = [5, 7, 9, 1.5, 1.5, 1.5, 0, 0];
#src θ0 = [6, 9, 2, 2, 0];

# Plot data pdf and initial model pdf
pf = plot(pdf(data_dis); xlims = (-1, 25), label="Gamma pdf",
    color = :black,
    xlabel = L"x",
    ylabel = L"p(x) \ \mathrm{ and } \ p(x;θ)",
)
plot!(pf, pdf(model(θ0)), label = "Initial Gaussian mixture", color=:blue)

#
prompt()


# ## Impractical score matching

lower = [fill(0, nmix); fill(1.0, nmix); fill(-Inf, nmix-1)]
upper = [fill(Inf, nmix); fill(Inf, nmix); fill(Inf, nmix-1)]
opt_sm = optimize(cost_sm1, lower, upper, θ0, Fminbox(BFGS()); autodiff = :forward)
θsm = minimizer(opt_sm)

plot!(pf, pdf(model(θsm)), label = "SM Gaussian mixture", color=:red)

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
plot!(ps, score(model(θsm)); label = "SM score function", color=:red)


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
opt_ml = optimize(negloglike, lower, upper, θsm, Fminbox(BFGS()); autodiff = :forward)
θml = minimizer(opt_ml)
negloglike.([θml, θsm, θ0])

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
## Practical score matching

The above SM fitting process
used `score(data_dis)`,
the score-function of the data distribution,
which is unknown in practical situations.

[Hyvärinen 2005](http://jmlr.org/papers/v6/hyvarinen05a.html)
derived the following more practical cost function
that is independent of the unknown data score function:
```math
J(\mathbf{θ}) =
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
=#


# Practical score-matching cost function
function cost_sp2(x::AbstractVector{<:Real}, θ)
    tmp = model(θ)
    model_score = score(tmp)
    return (1/T) * (sum(score_deriv(tmp), x) +
        0.5 * sum(abs2 ∘ model_score, x))
end;

cost_sp1 = (θ) -> cost_sp2(data, θ) # minimize this score-matching cost function
opt_sp = optimize(cost_sp1, lower, upper, θ0, Fminbox(BFGS()); autodiff = :forward)
θsp = minimizer(opt_sp)
cost_sm4.([θsp, θsm, θml])

#
plot!(pf, pdf(model(θsp)), label = "SP Gaussian mixture", color=:cyan)
plot!(ps, score(model(θsp)), label = "SP score function", color=:cyan)
pfs = plot(pf, ps)

#=
Curiously the supposedly equivalent SM cost function works much worse.
Like the ML estimate,
the first two ``σ`` values are stuck at the `lower` limit.
Could it be local extrema?
More investigation is needed!
=#


# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
