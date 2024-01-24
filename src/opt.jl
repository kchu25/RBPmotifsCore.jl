function prep_filters(D, hp, projs) 
    D_init      = D .^ 2  
    D_init_r    = reshape(D_init, (4, hp.filter_len, hp.M))
    D_init_r    += projs.pseudocount_matrix               # pseudocounts to avoid division by zero
    D_init_r    = D_init_r ./ sum(D_init_r, dims=1)
    D_init      = reshape(D_init_r, (hp.f_len, 1, hp.M))
    return D_init
end

function prep_syntax_filters(F)
    F = F.^2
    return F ./ (sqrt.(sum(F.^2, dims=(1,2)))) # normalize F
end

function prep_params(ucdl, hp, projs)
    lambda_sparsity_warmup  = ucdl.lambda_sparsity_warmup^2
    lambda_sparsity         = ucdl.lambda_sparsity.^2
    lambda_stepsize_warmup  = ucdl.lambda_stepsize_warmup^2
    omega_stepsize_warmup   = ucdl.omega_stepsize_warmup^2
    lambda_stepsize         = ucdl.lambda_stepsize.^2
    omega_stepsize          = ucdl.omega_stepsize.^2
    D                       = prep_filters(ucdl.D, hp, projs)
    F                       = prep_syntax_filters(ucdl.F)
    return lambda_sparsity_warmup, lambda_sparsity, 
           lambda_stepsize_warmup, omega_stepsize_warmup, 
           lambda_stepsize, omega_stepsize,
           D, F
end

function warmup_Z(S, D, lambda_stepsize_warmup, lambda_sparsity_warmup, projs)
    DᵀS             = convolution(S, D, pad=0, flipped=true)
    Z_update        = (lambda_stepsize_warmup .* DᵀS) .- (lambda_sparsity_warmup * lambda_stepsize_warmup)
    Z               = Flux.NNlib.relu.(projs.z_mask_n .* Z_update)
    return Z
end

function generate_bitmat(X, hp)
    bitmat      = CUDA.zeros(eltype(X), (size(X, 1), 1, hp.K, hp.batch_size))
    X_reshape   = reshape(X, (size(X, 1)*hp.K, hp.batch_size))
    vals        = reshape(partialsort.(eachcol(X_reshape), hp.q, rev=true) |> cu, (1,1,1, hp.batch_size))
    bitmat[X .≥ vals] .= 1;
    return bitmat
end

function project_X(X, hp)
    bitmat = @ignore generate_bitmat(X, hp)
    return X .* bitmat
end

function warmup_X(F, Z, omega_stepsize_warmup, hp, len)
    Z_reshaped = hp.magnifying_factor .* reshape(Z[1:4:end,:,:], (len.c, hp.M, 1, hp.batch_size))
    X_updated = 
        omega_stepsize_warmup .* convolution(Z_reshaped, F, pad=0, flipped=true) # (len.l, 1, hp.K, hp.batch_size)
    return project_X(X_updated, hp)
end

function warmup_XZ(S, D, F, lambda_stepsize_warmup, lambda_sparsity_warmup, 
                    omega_stepsize_warmup, hp, len, projs
                    )
    Z  = warmup_Z(S, D, lambda_stepsize_warmup, lambda_sparsity_warmup, projs)
    X  = warmup_X(F, Z, omega_stepsize_warmup, hp, len)
    FX = sum(convolution(X, F, pad=(hp.h-1, hp.M-1), groups=hp.K), dims=3) # TODO fix
    return Z, X, FX
end

function update_Z(S, Z, D, lambda_sparsity, lambda_stepsize, hp, projs, num_pass)
    ZD          = convolution(Z, D, pad=hp.f_len-1, groups=hp.M)
    diff        = sum(ZD, dims=2) - S
    z_grad      = convolution(diff, D, pad=0, flipped=true)
    Z_updated   = Z - lambda_stepsize[num_pass] .* z_grad .- (lambda_sparsity[num_pass] .* lambda_stepsize[num_pass])
    return Flux.NNlib.relu.(projs.z_mask_n .* Z_updated)
end

function update_X(FX, Z, X, F, omega_stepsize, hp, len, num_pass)
    Z_reshaped = hp.magnifying_factor .* reshape(Z[1:4:end,:,:], (len.c, hp.M, 1, hp.batch_size))
    diff        = sum(FX, dims=3) - Z_reshaped
    x_grad      = convolution(diff, F, pad=0, flipped=true)
    X_updated   = X - omega_stepsize[num_pass] .* x_grad
    return project_X(X_updated, hp)
end

function one_forward_step_XZ(S, Z, D, X, F, FX,
                              lambda_sparsity, lambda_stepsize, 
                              omega_stepsize, hp, len, projs, num_pass
                              )
    Z                   = update_Z(S, Z, D, lambda_sparsity, lambda_stepsize, hp, projs, num_pass)
    X                   = update_X(FX, Z, X, F, omega_stepsize, hp, len, num_pass)
    FX                  = sum(convolution(X, F, pad=(hp.h-1, hp.M-1), groups=hp.K), dims=3)
    return Z, X, FX
end

function loss(S, Z, X, D, F, hp)
    normalize_factor = (1.0f0/float_type(hp.batch_size));
    DZ                          = sum(convolution(Z, D, pad=hp.f_len-1, groups=hp.M), dims=2)
    reconstruction_loss         = normalize_factor*sum((DZ - S).^2)
    FX                          = sum(convolution(X, F, pad=(hp.h-1, hp.M-1), groups=hp.K), dims=3)
    Z_reshaped = hp.magnifying_factor .* reshape(Z[1:4:end,:,:], (len.c, hp.M, 1, hp.batch_size))

    syntax_reconstruction_loss  = normalize_factor*sum((FX - Z_reshaped).^2)
    return reconstruction_loss + syntax_reconstruction_loss
end


function ADMM_XYZ(S, D, F,  
                 lambda_stepsize_warmup, lambda_stepsize, 
                 lambda_sparsity_warmup, lambda_sparsity,
                 omega_stepsize_warmup, omega_stepsize, 
                 hp, len, projs
                 )

    # warm up
    Z, X, FX = 
        warmup_XZ(S, D, F, 
                   lambda_stepsize_warmup, lambda_sparsity_warmup, 
                   omega_stepsize_warmup, hp, len, projs)

    # iterations
    for num_pass = 1:hp.num_pass_xyz
        Z, X, FX = 
            one_forward_step_XZ(S, Z, D, X, F, FX,
                                 lambda_sparsity, lambda_stepsize, 
                                 omega_stepsize, hp, len, projs, num_pass)
    end
    return Z, X
end

function forward_pass_return_loss(S, cdl, hp, len, projs)
    lambda_sparsity_warmup, lambda_sparsity,
    lambda_stepsize_warmup, omega_stepsize_warmup, lambda_stepsize, omega_stepsize,
    D, F = prep_params(cdl, hp, projs)

    Z, X = ADMM_XYZ(S, D, F,  
                 lambda_stepsize_warmup, lambda_stepsize, 
                 lambda_sparsity_warmup, lambda_sparsity,
                 omega_stepsize_warmup, omega_stepsize, 
                 hp, len, projs
                 )

    l = loss(S, Z, X, D, F, hp)
    println("loss $l")
    return l
end

function retrieve_code(S, cdl, hp, len, projs)
    lambda_sparsity_warmup, lambda_sparsity,
    lambda_stepsize_warmup, omega_stepsize_warmup, lambda_stepsize, omega_stepsize,
    D, F_orig = prep_params(cdl, hp, projs)

    Z, X = ADMM_XYZ(S, D, F_orig,  
                 lambda_stepsize_warmup, lambda_stepsize, 
                 lambda_sparsity_warmup, lambda_sparsity,
                 omega_stepsize_warmup, omega_stepsize, 
                 hp, len, projs
                 )
    return F_orig, Z, X
end
