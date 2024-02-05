#=
# [Score Matching overview](@id 01-overview)

This page illustrates the Julia package
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
using InteractiveUtils: versioninfo


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
``\mathrm{θ}``
of a model distribution
``p(\mathrm{x}; \mathrm{θ})``
that "best fit" the data.

Maximum-likelihood estimation
is impractical for complicated models
where the normalizing constant is intractable.

[Hyväarinen 2005](http://jmlr.org/papers/v6/hyvarinen05a.html)
proposed an alternative called
_score matching_
that circumvents
needing to find the normalizing constant.

The idea of the score matching approach
is 
```math
\hat{\mathbf{θ}} = \arg \min_{\mathbf{θ}}
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
=#


# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
