module ConvolutionOperators

using LinearAlgebra


abstract type AbstractConvOp end

function Base.size(x::AbstractConvOp) map(length, axes(x)) end
function Base.size(x::AbstractConvOp, i) size(x)[i] end
function Base.axes(x::AbstractConvOp) map(Base.OneTo, size(x)) end
function Base.axes(x::AbstractConvOp, i) axes(x)[i] end

function timeslice(Z,k)
    T = eltype(Z)
    Zk = zeros(T, size(Z)[1:2])
    ConvolutionOperators.timeslice!(Zk, Z, k)
end

include("companion.jl")

include("zeroconvop.jl")
include("denseconvop.jl")
include("bandedconvop.jl")
include("linearcombinations.jl")
include("leftrightmul.jl")
include("liftedconvop.jl")
include("truncatedconvop.jl")
include("matrixconvop.jl")


end
