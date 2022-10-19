using LinearAlgebra

struct LeftRightMulCVO <: AbstractConvOp
    convop
    left_linear_map
    right_linear_map
end

function Base.size(x::LeftRightMulCVO)

    A = x.left_linear_map
    B = x.right_linear_map
    C = x.convop    

    sz1 = (A isa UniformScaling) ? size(C,1) : size(A,1)
    sz2 = (B isa UniformScaling) ? size(C,2) : size(B,2) 
    sz3 = size(C,3)
    return (sz1,sz2,sz3)
end

function Base.size(x::LeftRightMulCVO,i) size(x)[i] end

function Base.axes(x::LeftRightMulCVO) map(Base.OneTo, size(x)) end
function Base.axes(x::LeftRightMulCVO, i) axes(x)[i] end

function Base.:*(C::AbstractConvOp, B) LeftRightMulCVO(C,I,B) end
function Base.:*(A, C::AbstractConvOp) LeftRightMulCVO(C,A,I) end

function convolve!(y, Z::LeftRightMulCVO, x, X, j, k_start=1, k_stop=size(Z,3))

    CVO = Z.convop
    A = Z.left_linear_map
    B = Z.right_linear_map

    Bx = B * x
    BX = B * X

    T = promote_type(eltype(Z), eltype(Bx))
    ax1 = (Z.left_linear_map isa UniformScaling) ? axes(Z.convop,1) : axes(Z.left_linear_map,1)
    w = similar(y, ax1)

    fill!(w,0)
    convolve!(w, CVO, Bx, BX, j, k_start, k_stop)
    y .+= A*w
end