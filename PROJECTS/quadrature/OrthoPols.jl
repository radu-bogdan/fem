module OrthoPols
export ortho2eva3

function kjacopols2(x::T, a::T, b::T, n::Int) where T
# function kjacopols2(x::T, a::T, b::T, n::Int) where T <: AbstractFloat
    pols = zeros(T, n+1)
    ders = zeros(T, n+1)

    pkp1 = one(T)
    pols[1] = pkp1

    dkp1 = zero(T)
    ders[1] = dkp1

    if n == 0
        return pols, ders
    end

    pk = pkp1
    pkp1 = (a / 2 - b / 2) + (1 + a / 2 + b / 2) * x
    pols[2] = pkp1

    dk = dkp1
    dkp1 = (1 + a / 2 + b / 2)
    ders[2] = dkp1

    if n == 1
        return pols, ders
    end

    for k in 2:n
        pkm1 = pk
        pk = pkp1
        dkm1 = dk
        dk = dkp1

        alpha1 = (2 * k + a + b - 1) * (a * a - b * b)
        alpha2 = (2 * k + a + b - 1) * 
                 ((2 * k + a + b - 2) * (2 * k + a + b))
        beta = 2 * (k + a - 1) * (k + b - 1) * 
               (2 * k + a + b)
        gamma = (2 * k * (k + a + b) * (2 * k + a + b - 2))
        
        pkp1 = ((alpha1 + alpha2 * x) * pk - beta * pkm1) / gamma
        dkp1 = ((alpha1 + alpha2 * x) * dk - 
                beta * dkm1 + alpha2 * pk) / gamma

        pols[k+1] = pkp1
        ders[k+1] = dkp1
    end

    return pols, ders
end

function klegeypols3(x::T, y::T, n::Int) where T
# function klegeypols3(x::T, y::T, n::Int) where T <: AbstractFloat
    pols = zeros(T, n + 1)
    dersx = zeros(T, n + 1)
    dersy = zeros(T, n + 1)

    pkp1 = one(T)
    pols[1] = pkp1
    dkp1 = zero(T)
    dersx[1] = dkp1
    ykp1 = zero(T)
    dersy[1] = ykp1

    if n == 0
        return pols, dersx, dersy
    end

    pk = pkp1
    pkp1 = x
    pols[2] = pkp1
    dk = dkp1
    dkp1 = one(T)
    dersx[2] = dkp1
    yk = ykp1
    ykp1 = zero(T)
    dersy[2] = ykp1

    if n == 1
        return pols, dersx, dersy
    end

    for k in 1:n-1
        pkm1 = pk
        pk = pkp1
        dkm1 = dk
        dk = dkp1
        ykm1 = yk
        yk = ykp1
        pkp1 = ((2 * k + 1) * x * pk - k * pkm1 * y * y) / (k + 1)
        dkp1 = ((2 * k + 1) * (x * dk + pk) - k * dkm1 * y * y) / (k + 1)
        ykp1 = ((2 * k + 1) * (x * yk) - k * (pkm1 * 2 * y + ykm1 * y * y)) / (k + 1)
        pols[k+2] = pkp1
        dersx[k+2] = dkp1
        dersy[k+2] = ykp1
    end

    return pols, dersx, dersy
end

function ortho2eva30(mmax::Int, z::Vector{T}) where T
# function ortho2eva30(mmax::Int, z::Vector{T}) where T <: AbstractFloat
    # Constants
    # zero_T = zero(T)
    # sqrt2 = sqrt(T(2))
    # sqrt3 = sqrt(T(3))
    # r11 = T(-1) / 3
    # r12 = T(-1) / sqrt3
    # r21 = T(-1) / 3
    # r22 = T(2) / sqrt3
    zero_T = zero(T)
    sqrt2 = sqrt(2one(T))
    sqrt3 = sqrt(3one(T))
    r11 = -one(T) / 3
    r12 = -one(T) / sqrt3
    r21 = -one(T) / 3
    r22 = 2one(T) / sqrt3

    x, y = z

    # Map the reference triangle to the right triangle with vertices (-1,-1), (1,-1), (-1,1)
    a = r11 + r12 * y + x
    b = r21 + r22 * y

    # Evaluate the Koornwinder's polynomials via the three term recursion
    par1 = (2 * a + 1 + b) / 2
    par2 = (1 - b) / 2
    f1, f3, f4 = klegeypols3(par1, par2, mmax)

    f2 = zeros(T, mmax+1, mmax+1)
    f5 = zeros(T, mmax+1, mmax+1)
    for m in 0:mmax
        par1 = T(2 * m + 1)
        temp_f2, temp_f5 = kjacopols2(b, par1, zero_T, mmax - m)
        f2[1:length(temp_f2), m+1] = temp_f2
        f5[1:length(temp_f5), m+1] = temp_f5
    end

    pols = zeros(T, div((mmax + 1) * (mmax + 2), 2))
    dersx = zeros(T, div((mmax + 1) * (mmax + 2), 2))
    dersy = zeros(T, div((mmax + 1) * (mmax + 2), 2))

    kk = 0
    for m in 0:mmax
        for n in 0:m
            kk += 1
            # Evaluate the polynomial (m-n, n), and their derivatives with respect to x,y
            pols[kk] = f1[m-n+1] * f2[n+1, m-n+1]

            dersx[kk] = f3[m-n+1] * f2[n+1, m-n+1]

            dersy[kk] = f1[m-n+1] * f5[n+1, m-n+1] * r22 +
                        f3[m-n+1] * f2[n+1, m-n+1] * (r12 + r22 / 2) +
                        f4[m-n+1] * f2[n+1, m-n+1] * (-r22 / 2)

            # Normalize
            scale = sqrt(((1 + (m - n) + n) * (1 + (m - n) + (m - n))) / sqrt3)
            pols[kk] *= scale
            dersx[kk] *= scale
            dersy[kk] *= scale
        end
    end

    return pols, dersx, dersy
end

function ortho2eva3(mmax::Int, z::Vector{T}) where T
# function ortho2eva3(mmax::Int, z::Vector{T}) where T <: AbstractFloat
    # Check if z has exactly 2 elements
    @assert length(z) == 2 "z must be a vector of length 2"

    # Constants
    # c0 = T(1) / sqrt(T(3)) * sqrt(sqrt(T(3)))
    # c1 = sqrt(T(2)) * sqrt(sqrt(T(3)))
    # c2 = sqrt(T(2)) * sqrt(sqrt(T(3)))
    c0 = one(T) / sqrt(3one(T)) * sqrt(sqrt(3one(T)))
    c1 = sqrt(2one(T)) * sqrt(sqrt(3one(T)))
    c2 = sqrt(2one(T)) * sqrt(sqrt(3one(T)))

    if mmax == 0
        pols = [c0]
        dersx = [zero(T)]
        dersy = [zero(T)]
    elseif mmax == 1
        pols = [c0, z[1] * c1, z[2] * c2]
        dersx = [zero(T), c1, zero(T)]
        dersy = [zero(T), zero(T), c2]
    else
        pols, dersx, dersy = ortho2eva30(mmax, z)
    end

    return pols, dersx, dersy
end



end




mmax = 2
# z0 = [0, 0]
# z1 = [1, 0]
# z2 = [0, 1]

z0 = BigFloat.([0, 0])
z1 = BigFloat.([1, 0])
z2 = BigFloat.([0, 1])

# pols0, dersx0, dersy0 = ortho2eva3(mmax, z0)
# pols1, dersx1, dersy1 = ortho2eva3(mmax, z1)
# pols2, dersx2, dersy2 = ortho2eva3(mmax, z2)

# println(pols0)
# println(pols1)
# println(pols2)