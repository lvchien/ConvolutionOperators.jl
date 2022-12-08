struct ZeroConvOp{I,J} <: AbstractConvOp
    row_axis::I
    col_axis::J
end

function Base.eltype(x::ZeroConvOp)
    return Bool
end

function Base.axes(x::ZeroConvOp)
    (x.row_axis, x.col_axis, Base.OneTo(0))
end

function _getindex(x::ZeroConvOp, i, j, k) false end
function convolve!(y, Z::ZeroConvOp, x, X, j, k_start=1, k_stop=size(Z,3))
    fill!(y, false)
end

function timeslice!(Y, Z::ZeroConvOp, k)
    fill!(Y, false)
end

function tailindex(Z::ZeroConvOp)
    return 0
end

function hastail(Z::ZeroConvOp)
    return false
end