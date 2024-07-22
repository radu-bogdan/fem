function kjacopols2(x, a, b, n)
    pols = zeros(Float64, n+1)
    ders = zeros(Float64, n+1)

    pkp1 = 1.0
    pols[1] = pkp1

    dkp1 = 0.0
    ders[1] = dkp1

    if n == 0
        return pols, ders
    end

    pk = pkp1
    pkp1 = (a / 2.0 - b / 2.0) + (1.0 + a / 2.0 + b / 2.0) * x
    pols[2] = pkp1

    dk = dkp1
    dkp1 = (1.0 + a / 2.0 + b / 2.0)
    ders[2] = dkp1

    if n == 1
        return pols, ders
    end

    for k in 2:n
        pkm1 = pk
        pk = pkp1
        dkm1 = dk
        dk = dkp1

        alpha1 = (2.0 * k + a + b - 1.0) * (a * a - b * b)
        alpha2 = (2.0 * k + a + b - 1.0) * 
                 ((2.0 * k + a + b - 2.0) * (2.0 * k + a + b))
        beta = 2.0 * (k + a - 1.0) * (k + b - 1.0) * 
               (2.0 * k + a + b)
        gamma = (2.0 * k * (k + a + b) * (2.0 * k + a + b - 2.0))
        
        pkp1 = ((alpha1 + alpha2 * x) * pk - beta * pkm1) / gamma
        dkp1 = ((alpha1 + alpha2 * x) * dk - 
                beta * dkm1 + alpha2 * pk) / gamma

        pols[k+1] = pkp1
        ders[k+1] = dkp1
    end

    return pols, ders
end


function klegeypols3(x, y, n)
    pols = zeros(Float64, n + 1)
    dersx = zeros(Float64, n + 1)
    dersy = zeros(Float64, n + 1)

    pkp1 = 1.0
    pols[1] = pkp1
    dkp1 = 0.0
    dersx[1] = dkp1
    ykp1 = 0.0
    dersy[1] = ykp1

    if n == 0
        return pols, dersx, dersy
    end

    pk = pkp1
    pkp1 = x
    pols[2] = pkp1
    dk = dkp1
    dkp1 = 1.0
    dersx[2] = dkp1
    yk = ykp1
    ykp1 = 0.0
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
        pkp1 = ((2.0 * k + 1.0) * x * pk - k * pkm1 * y * y) / (k + 1.0)
        dkp1 = ((2.0 * k + 1.0) * (x * dk + pk) - k * dkm1 * y * y) / (k + 1.0)
        ykp1 = ((2.0 * k + 1.0) * (x * yk) - k * (pkm1 * 2.0 * y + ykm1 * y * y)) / (k + 1.0)
        pols[k+2] = pkp1
        dersx[k+2] = dkp1
        dersy[k+2] = ykp1
    end

    return pols, dersx, dersy
end



function ortho2eva30(mmax, z)
    # Constants
    zero = 0.0
    sqrt2 = sqrt(2.0)
    sqrt3 = sqrt(3.0)
    r11 = -1.0 / 3.0
    r12 = -1.0 / sqrt3
    r21 = -1.0 / 3.0
    r22 = 2.0 / sqrt3

    x, y = z

    # Map the reference triangle to the right triangle with vertices (-1,-1), (1,-1), (-1,1)
    a = r11 + r12 * y + x
    b = r21 + r22 * y

    # Evaluate the Koornwinder's polynomials via the three term recursion
    par1 = (2.0 * a + 1.0 + b) / 2.0
    par2 = (1.0 - b) / 2.0
    f1, f3, f4 = klegeypols3(par1, par2, mmax)

    f2 = zeros(Float64, mmax+1, mmax+1)
    f5 = zeros(Float64, mmax+1, mmax+1)
    for m in 0:mmax
        par1 = 2 * m + 1
        temp_f2, temp_f5 = kjacopols2(b, par1, zero, mmax - m)
        f2[1:length(temp_f2), m+1] = temp_f2
        f5[1:length(temp_f5), m+1] = temp_f5
    end

    pols = zeros(Float64, div((mmax + 1) * (mmax + 2), 2))
    dersx = zeros(Float64, div((mmax + 1) * (mmax + 2), 2))
    dersy = zeros(Float64, div((mmax + 1) * (mmax + 2), 2))

    kk = 0
    for m in 0:mmax
        for n in 0:m
            kk += 1
            # Evaluate the polynomial (m-n, n), and their derivatives with respect to x,y
            pols[kk] = f1[m-n+1] * f2[n+1, m-n+1]

            dersx[kk] = f3[m-n+1] * f2[n+1, m-n+1]

            dersy[kk] = f1[m-n+1] * f5[n+1, m-n+1] * r22 +
                        f3[m-n+1] * f2[n+1, m-n+1] * (r12 + r22 / 2.0) +
                        f4[m-n+1] * f2[n+1, m-n+1] * (-r22 / 2.0)

            # Normalize
            scale = sqrt(((1 + (m - n) + n) * (1 + (m - n) + (m - n))) / sqrt3)
            pols[kk] *= scale
            dersx[kk] *= scale
            dersy[kk] *= scale
        end
    end

    return pols, dersx, dersy
end




function ortho2eva3(mmax, z)
    # Check if z has exactly 2 elements
    @assert length(z) == 2 "z must be a vector of length 2"

    # Constants
    c0 = 1.0 / sqrt(3.0) * sqrt(sqrt(3.0))
    c1 = sqrt(2.0) * sqrt(sqrt(3.0))
    c2 = sqrt(2.0) * sqrt(sqrt(3.0))

    if mmax == 0
        pols = [c0]
        dersx = [0.0]
        dersy = [0.0]
    elseif mmax == 1
        pols = [c0, z[1] * c1, z[2] * c2]
        dersx = [0.0, c1, 0.0]
        dersy = [0.0, 0.0, c2]
    else
        pols, dersx, dersy = ortho2eva30(mmax, z)
    end

    return pols, dersx, dersy
end


mmax = 2
z0 = [0, 0]
z1 = [1, 0]
z2 = [0, 1]

pols0, dersx0, dersy0 = ortho2eva3(mmax, z0)
pols1, dersx1, dersy1 = ortho2eva3(mmax, z1)
pols2, dersx2, dersy2 = ortho2eva3(mmax, z2)

println(pols0)
println(pols1)
println(pols2)