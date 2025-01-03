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
    M, N, K = 0, 0, 0
    for i in 1:size(mat, 1)
        M += size(mat[i, 1], 1)
        for j in 1:size(mat, 2)
            if i == 1 
                N += size(mat[1, j], 2) 
            end
            K = max(K, size(mat[i, j], 3))
        end
    end
    return (M, N, K)
end

function convolve!(y, Z::MatrixConvOp, x, X, j, k_start=1, k_stop=size(Z,3))
    mat = Z.matconvop
    M0 = 0
    for i1 in 1:size(mat, 1)
        N0 = 0
        for i2 in 1:size(mat, 2)
            M, N = size(mat[i1, i2])[1:2]
            xi2 = x[N0+1:N0+N, :]
            Xi2 = X[N0+1:N0+N, :]
            Y = zeros(M)
            convolve!(Y, mat[i1, i2], xi2, Xi2, j, k_start, k_stop)
            y[M0+1:M0+M] .+= Y
            N0 += N
        end 
        M0 += size(mat[i1, 1], 1)
    end
    return y
end

function timeslice!(Y, Z::MatrixConvOp, k)
    fill!(Y, 0)
    mat = Z.matconvop
    M0 = 0
    for i in 1:size(mat, 1)
        N0 = 0
        for j in 1:size(mat, 2)
            slice = timeslice(mat[i, j], k)
            M, N = size(slice)
            Y[M0+1:M0+M, N0+1:N0+N] .= slice
            N0 += N
        end
        M0 += size(mat[i, 1], 1)
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