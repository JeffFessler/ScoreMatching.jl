#=
# [Score Matching overview](@id 01-overview)

This page introduces the Julia package
[`ScoreMatching`](https://github.com/JeffFessler/ScoreMatching.jl).
=#

#srcURL


# ### Setup

# Packages needed here.

using ADTypes: AutoForwardDiff
using Distributions: Distribution, Normal, MixtureModel, logpdf, pdf
using Distributions: Gamma, Uniform
import Distributions: logpdf, pdf
import Flux
using Flux: Chain, Dense, Adam
import ForwardDiff
using InteractiveUtils: versioninfo
using LinearAlgebra: tr, norm
using LaTeXStrings
using MIRTjim: jim, prompt
using Optim: optimize, BFGS, Fminbox
import Optim: minimizer
import Plots
using Plots: Plot, plot, plot!, scatter, histogram, quiver!, default, gui
using Plots.PlotMeasures: px
using Random: shuffle, seed!; seed!(0)
#src import ReverseDiff
#src using ScoreMatching
using StatsBase: mean, std

default(label="", markerstrokecolor=:auto)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:prompt);


#=
## Overview

Given ``T``
[IID](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
training data samples
``\bm{x}_1, …, \bm{x}_T ∈ \mathbb{R}^N``,
we often want to find the parameters
``\bm{θ}``
of a model distribution
``p(\bm{x}; \bm{θ})``
that "best fit" the data.

Maximum-likelihood estimation
is impractical for complicated models
where the normalizing constant is intractable.

[Hyvärinen 2005](http://jmlr.org/papers/v6/hyvarinen05a.html)
proposed an alternative called
_score matching_
that circumvents
needing to find the normalizing constant
by using the
_score function_
of the model distribution,
defined as
``
\bm{s}(\bm{x}; \bm{θ}) =
\nabla_{\bm{x}} \log p(\bm{x}; \bm{θ}).
``

## Score functions

Before describing score matching methods,
we first illustrate
what a score function looks like.

Consider the (improper) model
``
p(\bm{x}; \bm{θ}) = \frac{1}{Z(\bm{θ})} \mathrm{e}^{-β |x_2 - x_1|^α}
``
where here there are two parameters
``\bm{θ} = (β, α)``,
for ``β > 0`` and ``α > 1``.
The _score function_
for this model is
```math
\bm{s}(\bm{x}; \bm{θ})
=
\nabla_{\bm{x}} \log p(\bm{x}; \bm{θ})
=
- β \nabla_{\bm{x}} |x_2 - x_1|^α
= α β \begin{bmatrix} 1 \\ -1 \end{bmatrix}
|x_2 - x_1|^{α-1} \mathrm{sign}(x_2 - x_1).
```

This example is related to
generalized Gaussian image priors,
and,
for ``α=1``,
is related to total variation (TV) regularization.

Here is a visualization
of this 'TV' pdf
and the score functions.

The quiver plot
shows that the score function
describe directions that ascend the prior.
=#

function do_quiver!(p::Plot, x, y, dx, dy; thresh=0.02, scale=0.15)
    tmp = d -> maximum(abs, filter(!isnan, d))
    dmax = max(tmp(dx), tmp(dy))
    ix = 5:11:length(x)
    iy = 5:11:length(y)
    x, y = x .+ 0*y', 0*x .+ y'
    x = x[ix,iy]
    y = y[ix,iy]
    dx = dx[ix,iy] / dmax * scale
    dy = dy[ix,iy] / dmax * scale
    good = @. (abs(dx) > thresh) | (abs(dy) > thresh)
    x = x[good]
    y = y[good]
    dx = dx[good]
    dy = dy[good]
    Plots.arrow(:open, :head, 0.001, 0.001)
    return quiver!(p, x, y, quiver=(dx,dy);
        aspect_ratio = 1,
        title = "TV score quiver",
        color = :red,
    )
end;

if !@isdefined(ptv)
    α = 1.01 # fairly close to TV
    β = 1
    x1 = range(-1, 1, 101) * 2
    x2 = range(-1, 1, 101) * 2
    tv_pdf2 = @. exp(-β * abs(x2' - x1)^α) # ignoring partition constant
    ptv0 = jim(x1, x2, tv_pdf2; title = "'TV' pdf", clim = (0, 1),
        color=:cividis, xlabel = L"x_1", ylabel = L"x_2", prompt=false,
    )
    tv_score1 = @. β * abs(x2' - x1)^(α-1) * sign(x2' - x1)
    ptv1 = jim(x1, x2, tv_score1; title = "TV score₁", prompt=false,
        color=:cividis, xlabel = L"x_1", ylabel = L"x_2", clim = (-1,1) .* 1.2,
    )
    tv_score2 = @. -β * abs(x2' - x1)^(α-1) * sign(x2' - x1)
    ptv2 = jim(x1, x2, tv_score2; title = "TV score₂", prompt=false,
        color=:cividis, xlabel = L"x_1", ylabel = L"x_2", clim = (-1,1) .* 1.2,
    )
    ptvq = do_quiver!(deepcopy(ptv0), x1, x2, tv_score1, tv_score2)
    ptv = plot(ptv0, ptv1, ptvq, ptv2)
    ## Plots.savefig("score-tv.pdf")
end


#
prompt()


#=
## Score matching

The idea behind the score matching approach
to model fitting is
```math
\hat{\bm{θ}} = \arg \min_{\bm{θ}}
J(\bm{θ})
,\qquad
J(\bm{θ}) =
\frac{1}{T} ∑_{t=1}^T
\| \bm{s}(\bm{x}_t; \bm{θ}) - \bm{s}(\bm{x}_t) \|^2
```
where
``
\bm{s}(\bm{x}; \bm{θ}) =
\nabla_{\bm{x}} \log p(\bm{x}; \bm{θ})
``
is the _score function_
of the model distribution,
and
``
\bm{s}(\bm{x}) =
\nabla_{\bm{x}} \log p(\bm{x})
``
is the _score function_
of the (typically unknown) data distribution.

[Vincent, 2011](https://doi.org/10.1162/NECO_a_00142)
calls this approach
_explicit score matching_ (ESM).


## 1D example

We begin by illustrating the score function
for a simple
``\mathcal{N}(8, 3)``
distribution.
=#

# Some convenience methods
logpdf(d::Distribution) = x -> logpdf(d, x)
pdf(d::Distribution) = x -> pdf(d, x)
derivative(f::Function) = x -> ForwardDiff.derivative(f, x)
gradient(f::Function) = x -> ForwardDiff.gradient(f, x)
## hessian(f::Function) = x -> ForwardDiff.hessian(f, x)
score(d::Distribution) = derivative(logpdf(d))
score_deriv(d::Distribution) = derivative(score(d)); # scalar x only

gauss_μ = 8
gauss_σ = 3
gauss_disn = Expr(:call, :Normal, gauss_μ, gauss_σ)
gauss_dist = eval(gauss_disn)
xaxis = (L"x", (-1,1).*5gauss_σ .+ gauss_μ, (-3:3)*gauss_σ .+ gauss_μ)
left_margin = 20px; bottom_margin = 10px
pgp = plot(pdf(gauss_dist); label="$gauss_disn pdf", color = :blue,
    left_margin, bottom_margin,
    xaxis, yaxis = (L"p(x)", (0, 0.15), (0:3)*0.05), size=(600,200),
)

## Plots.savefig(pgp, "gauss-pdf.pdf")
prompt()

ylabel_score1 = L"s(x) = \frac{\mathrm{d}}{\mathrm{d}x} \, \log \ p(x)"
pgs = plot(derivative(logpdf(gauss_dist)); color=:red, xaxis, size=(600,200),
    label = "$gauss_disn score function",
    yaxis = (ylabel_score1, (-2,2), -3:3), left_margin, bottom_margin,
)

## Plots.savefig(pgs, "gauss-score.pdf")
prompt()

#src plot(pgp, pgs, layout=(2,1))


# Same plots for a gaussian mixture model (GMM)
mix = MixtureModel(Normal, [(2,1), (8,3), (16,2)], [0.3, 0.4, 0.3])
mix = MixtureModel(Normal, [(3,1), (13,3)], [0.4, 0.6])
xaxis = (L"x", (-4,24), [0, 3, 13, 20])
pmp = plot(pdf(mix); label="Gaussian mixture pdf", color = :blue,
    left_margin, bottom_margin, xaxis, size=(600,200),
    yaxis = (L"p(x)", (0, 0.17), (0:3)*0.05),
)

## Plots.savefig(pmp, "mix-pdf.pdf")
prompt()

pms = plot(derivative(logpdf(mix)); color=:red, xaxis, size=(600,200),
    label = "GMM score function",
    yaxis = (ylabel_score1, (-5,5), -4:2:4), left_margin, bottom_margin,
)

## Plots.savefig(pms, "mix-score.pdf")
prompt()

#src plot(pmp, pms)


#=
## Illustration

For didactic purposes,
we illustrate explicit score matching (ESM)
by fitting samples from a
[Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
to a mixture of gaussians.

=#


# Generate training data
if !@isdefined(data)
    T = 100
    gamma_k = 8 # shape
    gamma_θ = 1 # scale
    gamma_mode = gamma_k > 1 ? (gamma_k - 1) * gamma_θ : 0
    gamma_mean = gamma_k * gamma_θ
    gamma_std = sqrt(gamma_k) * gamma_θ
    data_disn = Expr(:call, :Gamma, gamma_k, gamma_θ)
    data_dis = eval(data_disn)
    data_score = derivative(logpdf(data_dis))
    data = Float32.(rand(data_dis, T))
    xlims = (-1, 25)
    xticks = [0, floor(Int, minimum(data)), gamma_mode, gamma_mean, ceil(Int, maximum(data))]
    xticks = sort(xticks) # ticks that span the data range

    pfd = scatter(data, zeros(T); xlims, xticks, color=:black)
    plot!(pfd, pdf(data_dis); label="$data_disn pdf",
        color = :black, xlabel = L"x", ylabel = L"p(x)")

    psd = plot(data_score; color=:black,
        label = "$(data_disn.args[1]) score function",
        xaxis=(L"x", (1,20), xticks), ylims = (-3, 5), yticks=[0,4],
    )
    psdn = deepcopy(psd)
    tmp = score(Normal(mean(data), std(data)))
    plot!(psdn, tmp; label = "Normal score function", line=:dash, color=:black)

    ph = histogram(data; linecolor=:blue,
        xlabel=L"x", size=(600,300), yaxis=("count", (0,15), 0:5:15),
        bins=-1:0.5:25, xlims, xticks, label="data histogram")
    ## Plots.savefig(ph, "gamma-data.pdf")
    plot!(ph, x -> T*0.5 * pdf(data_dis)(x);
        color=:black, label="$data_disn Distribution")
    ## Plots.savefig(ph, "gamma-fit.pdf")
end


if false # plots for a talk
    pdt = plot(pdf(data_dis); label="$data_disn pdf", color = :blue,
        left_margin, bottom_margin,
        xaxis = (L"x", xlims, xticks), ylabel = L"p(x)", size=(600,200))
    pst = deepcopy(psd)
    tmp = score(Normal(gamma_mean, gamma_std))
    plot!(pst, tmp; label = "Normal score function", size=(600,200), xlims,
        line=:dash, color=:magenta, left_margin, bottom_margin,
        ylabel=ylabel_score1)

    ## Plots.savefig(pdt, "gamma-pdf.pdf")
    ## Plots.savefig(pst, "gamma-score.pdf")
end


#=
To perform unconstrained minimization
of a ``D``-component mixture,
the following mapping from ``\mathbb{R}^{D-1}``
to the ``D``-dimensional simplex is helpful.
It is the inverse of the
[additive logratio transform](https://en.wikipedia.org/wiki/Compositional_data#Additive_logratio_transform).
It is related to the
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
pf = deepcopy(pfd)
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


# ## Explicit score matching (ESM) (impractical)

if !@isdefined(θesm)
    lower = [fill(0, nmix); fill(1.0, nmix); fill(-Inf, nmix-1)]
    upper = [fill(Inf, nmix); fill(Inf, nmix); fill(Inf, nmix-1)]
    opt_esm = optimize(cost_esm1, lower, upper, θ0, Fminbox(BFGS());
        autodiff = AutoForwardDiff())
    ## opt_esm = optimize(cost_esm1, θ0, BFGS(); autodiff = AutoForwardDiff()) # unconstrained
    θesm = minimizer(opt_esm)
end;

plot!(pf, pdf(model(θesm)), label = "ESM Gaussian mixture", color=:green)

#
prompt()


#=
Plot the data score and model score functions
to see how well they match.
The largest mismatch is in the tails of the distribution
where there are few (if any) data points.
=#
ps = deepcopy(psd)
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
opt_ml = optimize(negloglike, lower, upper, θ0, Fminbox(BFGS()); autodiff = AutoForwardDiff())
#src opt_ml = optimize(negloglike, θ0, BFGS(); autodiff = AutoForwardDiff())
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
## Implicit score matching (ISM) (more practical)

The above ESM fitting process
used `score(data_dis)`,
the score-function of the data distribution,
which is unknown in practical situations.

[Hyvärinen 2005](http://jmlr.org/papers/v6/hyvarinen05a.html)
derived the following more practical cost function
that is independent of the unknown data score function:
```math
J_{\mathrm{ISM}}(\bm{θ}) =
\frac{1}{T} ∑_{t=1}^T
∑_{i=1}^N ∂_i s_i(\bm{x}_t; \bm{θ})
 + \frac{1}{2} | s_i(\bm{x}_t; \bm{θ}) |^2,
```
ignoring a constant that is independent of ``θ,``
where
```math
∂_i s_i(\bm{x}; \bm{θ})
=
\frac{∂}{∂ x_i} s_i(\bm{x}; \bm{θ})
=
\frac{∂^2}{∂ x_i^2} \log p(\bm{x}; \bm{θ}).
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
if !@isdefined(θism)
    cost_ism1 = (θ) -> cost_ism2(data, θ)
    opt_ism = optimize(cost_ism1, lower, upper, θ0, Fminbox(BFGS()); autodiff = AutoForwardDiff())
    ##opt_ism = optimize(cost_ism1, θ0, BFGS(); autodiff = AutoForwardDiff())
    θism = minimizer(opt_ism)
    cost_ism1.([θism, θesm, θml])
end;

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

Ideally
(as ``T → ∞``),
the ESM and ISM cost functions
should differ by a constant independent of ``θ``.
Here they differ for small, finite ``T``.
=#

tmp = [θ0, θesm, θml, θism]
cost_esm1.(tmp) - cost_ism1.(tmp)


#=
## Regularized score matching (RSM)

[Kingma & LeCun, 2010](https://doi.org/10.5555/2997189.2997315)
reported some instability of ISM
and suggested a regularized version
corresponding to the following (practical) cost function:

```math
J_{\mathrm{RSM}}(\bm{θ}) =
J_{\mathrm{ISM}}(\bm{θ}) + λ R(\bm{θ})
,\quad
R(\bm{θ}) =
\frac{1}{T} ∑_{t=1}^T
∑_{i=1}^N | ∂_i s_i(\bm{x}_t; \bm{θ}) |^2.
```
=#


# Regularized score matching (RSM) cost function
function cost_rsm2(x::AbstractVector{<:Real}, θ, λ)
    mod = model(θ)
    model_score = score(mod)
    tmp = score_deriv(mod).(x)
    R = sum(abs2, tmp)
    J_ism = sum(tmp) + 0.5 * sum(abs2 ∘ model_score, x)
    return (1/T) * (J_ism + λ * R)
end;

# Minimize this RSM cost function:
λ = 4e-1
cost_rsm1 = (θ) -> cost_rsm2(data, θ, λ)

if !@isdefined(θrsm)
    opt_rsm = optimize(cost_rsm1, lower, upper, θ0, Fminbox(BFGS());
        autodiff = AutoForwardDiff())
    θrsm = minimizer(opt_rsm)
    cost_rsm1.([θrsm, θ0, θism, θesm, θml])
end

#
plot!(pf, pdf(model(θism)), label = "RSM Gaussian mixture", color=:red)
plot!(ps, score(model(θism)), label = "RSM score function", color=:red)
pfs = plot(pf, ps)

#
prompt()


#=
Sadly the regularized score matching (RSM) approach did not help much here.
Increasing ``λ`` led to `optimize` errors.


## Denoising score matching (DSM)

[Vincent, 2011](https://doi.org/10.1162/NECO_a_00142)
proposed a practical approach
called
_denoising score matching_ (DSM)
that matches
the model score function
to the score function
of a Parzen density estimate
of the form
```math
q_{σ}(\bm{x}) = \frac{1}{T} ∑_{t=1}^T g_{σ}(\bm{x} - \bm{x}_t)
```
where ``g_{σ}`` denotes a Gaussian distribution
``\mathcal{N}(\bm{0}, σ \bm{I})``.

Statistically,
this approach is equivalent
(in expectation)
to adding noise
to the measurements,
and then applying
the ESM approach.
The DSM cost function is
```math
J_{\mathrm{DSM}}(\bm{θ}) =
\frac{1}{T} ∑_{t=1}^T
\mathbb{E}_{\bm{z} ∼ g_{σ}}\left[
\frac{1}{2}
\left\|
\bm{s}(\bm{x}_t + \bm{z}; \bm{θ}) + \frac{\bm{z}}{σ^2}
\right\|_2^2
\right].
```

A benefit of this approach
is that it does not require
differentiating the model score function w.r.t ``\bm{x}``.

The inner expectation over ``g_{σ}``
is typically analytically intractable,
so in practice
we replace it with a sample mean
where we draw ``M ≥ 1`` values of ``\bm{z}``
per training sample,
leading to the following practical cost function
```math
J_{\mathrm{DSM}, \, M}(\bm{θ}) =
\frac{1}{T} ∑_{t=1}^T
\frac{1}{M} ∑_{m=1}^M
\frac{1}{2}
\left\|
s(\bm{x}_t + \bm{z}_{t,m}; \bm{θ}) + \frac{\bm{z}_{t,m}}{σ^2}
\right\|_2^2,
```
where the noise samples
``\bm{z}_{t,m}``
are IID.


The next code blocks investigate this DSM approach
for somewhat arbitrary choices of ``M`` and ``σ``.
=#

seed!(0)
M = 9
σdsm = 1.0
zdsm = σdsm * randn(T, M);

#=
Define denoising score-matching cost function,
where input `data` has size ``(T,)``
and input `z` has size ``(T,M)``.
=#
function cost_dsm2(data::AbstractVector{<:Real}, z::AbstractArray{<:Real}, θ)
    model_score = score(model(θ))
    tmp = model_score.(data .+ z) # (T,M) # add noise to data
    return (0.5/T/M) * sum(abs2, tmp + z ./ σdsm^2)
end;

if !@isdefined(θdsm)
    cost_dsm1 = (θ) -> cost_dsm2(data, zdsm, θ) # + β * 0.5 * norm(θ)^2;
    opt_dsm = optimize(cost_dsm1, lower, upper, θ0, Fminbox(BFGS());
        autodiff = AutoForwardDiff())
    θdsm = minimizer(opt_dsm)
end;

plot!(pf, pdf(model(θdsm)); label = "DSM Gaussian mixture", color=:orange)
plot!(ps, score(model(θdsm)); label = "DSM score function", color=:orange)
pfs = plot(pf, ps)

# At least for this case, DSM worked better than ML, ISM and RSM.
prompt()


#=
## Noise-conditional score-matching (NCSM)

Above we used a single noise value for DSM.
Contemporary methods use a range of noise values
with noise-conditional score models,
e.g.,
[Song et al. ICLR 2021](https://openreview.net/forum?id=PxTIG12RRHS).

A representative formulation
uses a ``σ^2``-weighted expectation
like the following:
```math
J_{\mathrm{NCSM}}(\bm{θ}) ≜
\mathbb{E}_{σ ∼ f(σ)}\left[ σ^2 J_{\mathrm{DSM}}(\bm{θ}, σ) \right],
```
```math
J_{\mathrm{DSM}}(\bm{θ}, σ) ≜
\frac{1}{T} ∑_{t=1}^T
\mathbb{E}_{\bm{z} ∼ g_{σ}}\left[
\frac{1}{2}
\left\|
\bm{s}(\bm{x}_t + \bm{z}; \bm{θ}, σ) + \frac{\bm{z}}{σ^2}
\right\|_2^2
\right].
```
for some distribution ``f(σ)`` of noise levels.

Here we use a multi-layer perceptron (MLP)
to model the 1D noise-conditional score function
``\bm{s}(\bm{x}; \bm{θ}, σ)``.
We use a residual approach
with data normalization
and the baseline score of a standard normal distribution.
=#

function make_nnmodel(
    data; # (2,?)
    nweight = [2, 8, 16, 8, 1], # 2 inputs: [x, σ] for NCSM
    μdata = mean(data),
    σdata = std(data),
)
    layers = [Dense(nweight[i-1], nweight[i], Flux.gelu) for i in 2:length(nweight)]
    ## Change of variables y ≜ (x - μdata) / σ(x + z)
    pick1(x) = transpose(x[1,:]) # extract 1D data as a row vector
    pick2(x) = transpose(x[2,:]) # extract σ as a row vector
    quad(σ) = sqrt(σ^2 + σdata^2) # σ(x + z) (quadrature)
    scale1(x) = [(pick1(x) .- μdata) ./ quad.(pick2(x)); pick2(x)] # standardize
    ## The baseline score function here is just -y; use in a "ResNet" way:
    tmp1 = Flux.Parallel(.-, Chain(layers...), pick1)
    scale2(σ) = quad.(σ) / σdata^2 # "un-standardize" for final score w.r.t x
    tmp2 = Flux.Parallel(.*, tmp1, scale2 ∘ pick2)
    return Chain(scale1, tmp2)
end;

# Function to make data pairs suitable for NN training.
# Each use of this function's output is akin to ``M`` epochs of `data`.
# Use `shuffle` in case we use mini-batches later.
function dsm_data(
    M::Int = 9,
    σdist = Uniform(0.2,2.0),
)
    σdsm = Float32.(rand(σdist, T, M))
    z = σdsm .* randn(Float32, T, M)
    tmp1 = shuffle(data) .+ z # (T,M)
    tmp2 = transpose([vec(tmp1) vec(σdsm)]) # (2, T*M)
    return (tmp2, -transpose(vec(z ./ σdsm.^2)))
end

if !@isdefined(nnmodel)
    nnmodel = make_nnmodel(data;)

#src loss3(model, x, y) = Flux.mse(model(x), y) / 2 # old
    ## σ^2-weighted MSE loss:
    loss3(model, x, y) = mean(abs2, (model(x) - y) .* transpose(x[2,:])) / 2

    iters = 2^9
    dataset = [dsm_data() for i in 1:iters]
    state1 = Flux.setup(Adam(), nnmodel)
    @info "begin train"
    @time Flux.train!(loss3, nnmodel, dataset, state1)
    @info "end train"
end

nnscore = x -> nnmodel([x, 0.4])[1] # lower end of σdist range
tmp = deepcopy(ps)
plot!(tmp, nnscore; label = "NN score function", color=:blue)

# The noise-conditional NN score model worked OK.
# Training "coarse to fine" might work better than the `Uniform` approach above.
prompt()

if false # look at the set of NN score functions
    tmp = deepcopy(psd)
    for s in 0.1:0.2:1.6
        plot!(tmp, x -> nnmodel([x, s])[1]; label = "NN score function $s")
    end
    gui()
end


# Plots of data distribution with various added noise levels

σlist = [0.0005, 0.05, 0.1, 0.5, 1, 5]
nσ = length(σlist)
pl = Array{Any}(undef, nσ)
for (i, σ) in enumerate(σlist)
    local mix = MixtureModel(Normal, [(d,σ) for d in data])
    xm = range(0, 20, step = min(σ/5, 0.1))
    local tmp = pdf(mix).(xm)
    pl[i] = plot(xm, tmp; color = :red,
        xaxis = (L"x", (0, 20), [0, gamma_mean, 18]),
        yaxis = (L"q_σ(x)", [0,]), title = "σ = $σ",
    )
    ## plot!(pl, xm, tmp / maximum(tmp), label = "σ = $σ")
end
plot(pl...)

## Plots.savefig("data-qsig.pdf")
prompt()


include("../../../inc/reproduce.jl")
