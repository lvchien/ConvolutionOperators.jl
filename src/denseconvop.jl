struct DenseConvOp{T}  <: AbstractConvOp
    data::AbstractArray{T,3}
end

function Base.size(x::DenseConvOp) size(x.data) end
function _getindex(x::DenseConvOp, i, j, k) getindex(x.data, i, j, k) end

function convolve!(y, Z::DenseConvOp, x, X, j, k_start=1, k_stop=size(Z,3))

    M,N,K = size(Z)
    # @show k_stop
    for k in k_start : min(j,k_stop,K)
        i = j - k + 1
        y .+= Z[:,:,k] * x[:,i]
    end
    return y
end

function timeslice!(Y, Z::DenseConvOp, k)
    @assert size(Y) == size(Z)[1:2]
    for m in axes(Z,1)
        for n in axes(Z,2)
            Y[m,n] += _getindex(Z, m,n,k)
    end end
    return Y 
end

struct DenseConvOpAsArray{T} <: AbstractArray{T,3}
    convop::DenseConvOp{T}
end

function Base.size(x::DenseConvOpAsArray) size(x.convop) end
function Base.getindex(x::DenseConvOpAsArray, I::Vararg{Int, 3}) _getindex(x,I...) end

function Base.AbstractArray(x::DenseConvOp{T}) where {T}
    DenseConvOpAsArray(x)
end