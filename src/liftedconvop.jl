struct LiftedConvOp <: AbstractConvOp
    convop
    rowaxis
    colaxis
    rowblock
    colblock
end

function Base.eltype(x::LiftedConvOp)
    return eltype(x.convop)
end

function Base.axes(x::LiftedConvOp)
    (x.rowaxis, x.colaxis, axes(x.convop,3))
end
function Base.axes(x::LiftedConvOp, i) axes(x)[i] end

function Base.size(x::LiftedConvOp) length.(axes(x)) end
function Base.size(x::LiftedConvOp,i)
    szs = size(x)
    return szs[i]
end

function convolve!(y, L::LiftedConvOp, x, X, j, k_start=1, k_stop=size(L,3))

    y = view(y, axes(L,1))
    x = view(x, axes(L,2), axes(x,2))
    X = view(X, axes(L,2), axes(X,2))

    I = L.rowblock
    J = L.colblock

    yI = view(y, I)
    xJ = view(x, J, axes(x,2))
    XJ = view(X, J, axes(x,2))
    LIJ = L.convop

    convolve!(yI, LIJ, xJ, XJ, j, k_start, k_stop)
end

function timeslice!(Y, L::LiftedConvOp, k)

    I = L.rowblock
    J = L.colblock
    LIJ = L.convop

    ax1, ax2, _ = axes(L)

    timeslice!(view(Y,ax1[I],ax2[J]),LIJ,k)
    return Y
end