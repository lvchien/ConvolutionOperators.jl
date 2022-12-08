struct TruncatedConvOp{T} <: AbstractConvOp
    convop
    kmax
end

function truncate(x::AbstractConvOp, kmax)
    TruncatedConvOp{eltype(x)}(x, kmax)
end

function Base.eltype(x::TruncatedConvOp)
    return eltype(x.convop)
end

function Base.size(x::TruncatedConvOp)
    size(x.convop)
    # ln = minimum(size(x.convop)[3], x.kmax)
    # return (size(x.convop)[1:2]..., ln)
end

function convolve!(y, Z::TruncatedConvOp, x, X, j, k_start=1, k_stop=size(Z,3))
    k_stop = min(k_stop, size(Z,3))
    convolve!(y, Z.convop, x, X, j, k_start, k_stop)
end

function timeslice!(Y, Z::TruncatedConvOp, k)
    k > Z.kmax && return Y
    timeslice!(Y, Z.convop, k)
end

function hastail(Z::TruncatedConvOp)
    false
end