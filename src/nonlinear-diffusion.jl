using ImageFiltering: Kernel, imfilter
using StaticArrays: SVector


# function nld_step_scalar(Ld, c, stepsize)
#     dx = 0.5 * stepsize * (c[1:end, 1:end-1] + c[1:end, 2:end]) .* (Ld[1:end, 2:end] - Ld[1:end, 1:end-1])
#     dy = 0.5 * stepsize * (c[1:end-1, 1:end] + c[2:end, 1:end]) .* (Ld[2:end, 1:end] - Ld[1:end-1, 1:end])

#     Ld[1:end, 1:end-1] .+= dx
#     Ld[1:end, 2:end] .-= dx
#     Ld[1:end-1, 1:end] .+= dy
#     Ld[2:end, 1:end] .-= dy
# end

function nld_step_scalar(Ld, c, stepsize, dx, dy) @inbounds begin
    rows::Int64=size(Ld,1)
    cols::Int64=size(Ld,2)

    # for k in 1:cols-1
    #     for j in 1:rows
    #         dx[j,k+1] = 0.5 * stepsize * (c[j,k] + c[j,k+1]) .* (Ld[j,k+1] - Ld[j,k])
    #     end
    # end
    # for k in 1:cols
    #     for j in 1:rows-1
    #         dy[j+1,k] = 0.5 * stepsize * (c[j,k] + c[j+1, k]) .* (Ld[j+1, k] - Ld[j,k])
    #     end
    # end

    for k in 1:cols-1
        for j in 1:rows-1
            cjk=c[j,k]
            Ldjk = Ld[j,k]
            dx[j,k+1] = 0.5 * stepsize * (cjk + c[j,k+1]) * (Ld[j,k+1] - Ldjk)
            dy[j+1,k] = 0.5 * stepsize * (cjk + c[j+1, k]) * (Ld[j+1, k] - Ldjk)
        end
    end
    j=rows
    for k in 1:cols-1
        dx[j,k+1] = 0.5 * stepsize * (c[j,k] + c[j,k+1]) .* (Ld[j,k+1] - Ld[j,k])
    end
    k=cols
    for j in 1:rows-1
        dy[j+1,k] = 0.5 * stepsize * (c[j,k] + c[j+1, k]) .* (Ld[j+1, k] - Ld[j,k])
    end

    for k in 1:cols
        for j in 1:rows
            Ld[j,k] += dx[j,k+1] - dx[j,k] + dy[j+1,k] - dy[j,k]
        end
    end
end end

pm_g1_diffusivity!(dst, Lx, Ly, k) =
    calculate_diffusivity′!(dst, Lx, Ly, k) do dL
        exp(-dL)
    end

pm_g2_diffusivity!(dst, Lx, Ly, k) =
    calculate_diffusivity′!(dst, Lx, Ly, k) do dL
        1.0 / (1.0 + dL)
    end

weickert_diffusivity!(dst, Lx, Ly, k) =
    calculate_diffusivity′!(dst, Lx, Ly, k) do dL
        1.0 - exp(-3.315 / (dL * dL * dL * dL))
    end

charbonnier_diffusivity!(dst, Lx, Ly, k) =
    calculate_diffusivity′!(dst, Lx, Ly, k) do dL
        1.0 / sqrt(1.0 + dL)
    end

pm_g1_diffusivity(Lx, Ly, k) = pm_g1_diffusivity!(similar(Lx), Lx, Ly, k)
pm_g2_diffusivity(Lx, Ly, k) = pm_g2_diffusivity!(similar(Lx), Lx, Ly, k)
weickert_diffusivity(Lx, Ly, k) = weickert_diffusivity!(similar(Lx), Lx, Ly, k)
charbonnier_diffusivity(Lx, Ly, k) = charbonnier_diffusivity!(similar(Lx), Lx, Ly, k)

function calculate_diffusivity′!(func, dst, Lx, Ly, contrast)
    invk2 = 1.0 / (contrast * contrast)
    @inbounds begin
        for k in axes(Lx, 2)
            @simd for j in axes(Lx, 1)
                lx, ly = Lx[j, k], Ly[j, k]
                dL = (lx * lx + ly * ly) * invk2
                dst[j, k] = func(dL)
            end
        end
    end
    dst
end

function compute_k_percentile(img, perc; gscale = 1.0, nbins = 300)
    Lsmooth = imfilter(img, Kernel.gaussian(gscale))
    compute_k_percentile_scharr′(Lsmooth, perc, nbins)
end

function compute_k_percentile_scharr′(L, perc, nbins) @inbounds begin
    rows, cols = size(L)

    hist = zeros(Int32, nbins)

    ## Interior pixels only: a 3x3 stencil on the border would need padding that
    ## never enters the histogram.
    modg = Matrix{Float64}(undef, rows-2, cols-2)
    hmax = 0.0
    for k in 2:cols-1
        @simd for j in 2:rows-1
            lx, ly = scharr32(L, j, k)
            m = sqrt(lx^2 + ly^2)
            modg[j-1, k-1] = m
            hmax = max(hmax, m)
        end
    end

    ## A constant image has no gradient to take a percentile of; the bin scale
    ## below would be `Inf`.
    hmax <= 1e-10 && return 0.03

    indexscale = nbins / hmax

    for m in modg
        ## `hmax * (nbins / hmax)` is not exactly `nbins` in floating point, so
        ## the top bin can round to `nbins + 1`.
        nbin = clamp(ceil(Int, m * indexscale), 1, nbins)
        if m > 1e-10
            hist[nbin] += 1
        end
    end

    nthreshold = floor(Int, sum(hist) * perc)
    bin = findfirst(hx -> hx > nthreshold, cumsum(hist))

    return if isnothing(bin)
        0.03
    else
        bin / indexscale
    end
end end

## Scharr factors scaled by 32, the single source of truth for the gradient
## operator. `scharr32_x`/`scharr32_y` filter whole images; `scharr32` evaluates
## the same operator one pixel at a time.
const scharr32_smooth = SVector(3.0, 10.0, 3.0) * 2
const scharr32_diff = SVector(-1.0, 0.0, 1.0) / 2
const scharr32_x = DilatedSeparable(scharr32_smooth, scharr32_diff, 1)
const scharr32_y = DilatedSeparable(scharr32_diff, scharr32_smooth, 1)

"""
    scharr32(L, j, k)

The 3×3 Scharr x- and y-gradients of `L` at row `j`, column `k`, scaled by 32 —
the same operator as [`scharr32_x`](@ref)/[`scharr32_y`](@ref), but with both
passes fused into one stencil evaluation so no intermediate image is needed.
Requires `(j, k)` to be an interior pixel.
"""
@inline function scharr32(L, j, k) @inbounds begin
    sm, s0, sp = scharr32_smooth
    _, _, dp = scharr32_diff
    a = L[j-1, k-1]; b = L[j-1, k]; c = L[j-1, k+1]
    d = L[j,   k-1];                f = L[j,   k+1]
    g = L[j+1, k-1]; h = L[j+1, k]; i = L[j+1, k+1]
    ## The separable kernels below, expanded: the smoothing factor weights the
    ## three lines across the derivative direction, the difference factor the two
    ## outer lines along it.
    lx = (sm*dp)*(c - a) + (s0*dp)*(f - d) + (sp*dp)*(i - g)
    ly = (sm*dp)*(g - a) + (s0*dp)*(h - b) + (sp*dp)*(i - c)
    (lx, ly)
end end


function demo_diffusivity_functions()
    imshow(Lsmooth)
    aa = pm_g1_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
    aa = pm_g2_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
    aa = weickert_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
    aa = charbonnier_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
end

function demo_k_percentile()
    gthresh = compute_k_percentile(img, 0.95)
    imshow(sqrt.(Lx .^ 2 + Ly .^ 2))
    imshow(sqrt.(Lx .^ 2 + Ly .^ 2) .< gthresh)
end

function demo_nld()
    Lflow = charbonnier_diffusivity(Lx, Ly, 0.01)

    imshow(copy(Lt))
    for _ = 1:3
        for _ = 1:100
            nld_step_scalar(Lt, Lflow, 0.05)
        end
        imshow(copy(Lt))
    end
end

function compute_derivative_kernels(scale)

    ## The dense kernel would be length 3 + 2*(scale-1) with its outer taps at
    ## ±(scale), so the dilation radius simply is the scale.
    w = 10.0/3.0
    norm = 1.0/(2.0*scale*(w+2.0))

    smooth = SVector(norm, w*norm, norm)
    central = SVector(-1.0, 0.0, 1.0)

    (DilatedSeparable(smooth, central, scale), DilatedSeparable(central, smooth, scale))
end
