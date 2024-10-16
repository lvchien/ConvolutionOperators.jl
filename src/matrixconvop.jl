struct MatrixConvOp <: AbstractConvOp
    matconvop::Matrix{AbstractConvOp}
end

function Base.eltype(Z::MatrixConvOp)
    mat = Z.matconvop
    type = eltype(mat[1, 1])
    for i in 1:size(mat, 1)
        for j in 1:size(mat, 2)
            type = promote_type(type, eltype(mat[i, j]))
        end
    end
    return type
end

function Base.size(Z::MatrixConvOp)
    mat = Z.matconvop
    blocksize = size(mat[1, 1])
    sz3 = 0
    for i in 1:size(mat, 1)
        for j in 1:size(mat, 2)
            sz3 = max(sz3, size(mat[i, j], 3))
        end
    end
    return (blocksize[1]*size(mat, 1), blocksize[2]*size(mat, 2), sz3)
end

function convolve!(y, Z::MatrixConvOp, x, X, j, k_start=1, k_stop=size(Z,3))
    mat = Z.matconvop
    blocksize = size(mat[1, 1])

    for i1 in 1:size(mat, 1)
        for i2 in 1:size(mat, 2)
            xi2 = x[(i2-1)*blocksize[2]+1:i2*blocksize[2], :]
            Xi2 = X[(i2-1)*blocksize[2]+1:i2*blocksize[2], :]
            Y = zeros(size(xi2)[1])
            convolve!(Y, mat[i1, i2], xi2, Xi2, j, k_start, k_stop)
            y[(i1-1)*blocksize[1]+1:i1*blocksize[1]] .+= Y
        end 
    end
    return y
end

function timeslice!(Y, Z::MatrixConvOp, k)
    fill!(Y, 0)
    mat = Z.matconvop
    slice = timeslice.(mat, k)
    blocksize = size(mat[1, 1])
    for i in 1:size(mat, 1)
        for j in 1:size(mat, 2)
            Y[(i-1)*blocksize[1]+1:i*blocksize[1], (j-1)*blocksize[2]+1:j*blocksize[2]] .= slice[i, j]
        end
    end
    return Y
end

function tailindex(Z::MatrixConvOp) 
    mat = Z.matconvop
    ti = 0
    for i in 1:size(mat, 1)
        for j in 1:size(mat, 2)
            ti = max(ti, tailindex(mat[i, j]))
        end
    end
    return ti
end

function hastail(Z::MatrixConvOp)
    mat = Z.matconvop
    for i in 1:size(mat, 1)
        for j in 1:size(mat, 2)
            if hastail(mat[i, j]) return true end
        end
    end
    false
end