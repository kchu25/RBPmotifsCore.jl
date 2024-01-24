
function randomly_initialize_filters(; 
                       dim=4,
                       rng=Random.GLOBAL_RNG, 
                       repeats=5, 
                       how_many_filters=10,
                       float_type=Float16)
    arr = zeros(float_type,(dim+1,
                            repeats,
                            1,
                            how_many_filters));
    for i = 1:repeats, j = 1:how_many_filters
        unif = rand(rng, dim-1);
        arr[2:dim,i,1,j] .= sort(unif);
    end
    arr[dim+1,:,:,:] .= 1;
    return reshape(diff(arr, dims=1), (dim*repeats,1,how_many_filters));
end

Base.@kwdef mutable struct Hyperparam
    filter_len::Int = 6                 # length of each filter
    f_len::Int = filter_len*4           # length of the filter adjusted for one hot DNA sequence
    M::Int = 24                         # number of filters
    h::Int = 12                         # height of the syntax filter
    K::Int = 24                         # number of syntax filters
    q::Int = 32                         # how sparse the syntax code should be for each sequence
    batch_size::Int = 32                 # batch size
    num_pass_xyz::Int = 4               # number of passes for the x,y,z parameters
    # num_pass_df::Int = 3                # number of passes for the d,f parameters
    magnifying_factor::float_type = 10  # magnifying factor for the sparse code Z, Y
    gamma::float_type = 0.1             # regularization for filter incoherence
end

struct length_info
    L::Int                              # length of the sequence
    C::Int                              # code length
    c::Int                              # code length divided by 4
    l::Int                              # syntax code length 
    MB::Int                             # hp.M * hp.batch_size
    KB::Int                             # hp.K * hp.batch_size
    CS_vlen::Int                        # C + L - 1
    last_avail_ind::Int                 # last data point in the code that can be used for a full batch

    function length_info(hp, data)
        L               = 4*data.L
        C               = L-hp.f_len+1
        c               = data.L-hp.filter_len+1
        l               = c-hp.h+1
        MB              = hp.M * hp.batch_size
        KB              = hp.K * hp.batch_size
        CS_vlen         = C + L - 1;
        last_avail_ind  = (data.N - data.N % hp.batch_size);
        new(L, C, c, l, MB, KB, CS_vlen, last_avail_ind)
    end
end

struct projectors
    mapdrange::CuArray{float_type, 2}
    mapclarge::CuArray{float_type, 2}
    z_mask_n::CuArray{float_type, 3, CUDA.Mem.DeviceBuffer}
    pseudocount_matrix::CuArray{float_type, 3}

    function projectors(hp, len)
        mapdrange                               = zeros(float_type, (hp.f_len, len.C+len.L-1));
        mapdrange[:, len.C:len.C+hp.f_len-1]    = Matrix(I, hp.f_len, hp.f_len);
        mapdrange                               = cu(mapdrange);

        mapclarge                               = zeros(float_type, (len.C, len.c));    
        mapclarge[1:4:end,:]                    = Matrix(I, len.c, len.c);
        mapclarge                               = cu(mapclarge);

        z_mask_col  = 1:len.C .∈ [1:4:len.C]; 
        z_mask_n    = cu(float_type.(repeat(z_mask_col, outer=(1,hp.M,hp.batch_size))));
        pseudocount_matrix = fill(0.001f0, (4, hp.filter_len, hp.M))

        new(
            mapdrange,
            mapclarge,
            z_mask_n,
            pseudocount_matrix
        )
    end
end

struct ucdl
    lambda_sparsity_warmup::float_type                  # sparsity param for Z and Y when their initial values are zero
    lambda_sparsity::Array{float_type, 1}               # sparsity param for Z and Y during iterations

    lambda_stepsize_warmup::float_type                  # step size for Z and Y when their initial values are zero
    omega_stepsize_warmup::float_type                   # step size for X when its initial values is zero
    lambda_stepsize::Array{float_type, 1}               # step size for Z and Y during iterations
    omega_stepsize::Array{float_type, 1}                # step size for X during iterations

    D::CuArray{float_type, 3}                           # filters (f_len, 1, M, K)    
    F::CuArray{float_type, 4}                           # syntax filters (hp.h, hp.twoM, 1, hp.K)
   
    function ucdl(hp; η₁=float_type(0.05))
        D                       = cu(randomly_initialize_filters(
                                   repeats=hp.filter_len, 
                                   how_many_filters=hp.M,
                                   float_type=float_type));
        D                       = sqrt.(D);
        F = cu(abs.(0.1 .* randn(float_type, (hp.h, hp.M, 1, hp.K))));
        lambda_sparsity_warmup  = η₁ * rand(float_type);
        lambda_sparsity         = η₁ * rand(float_type, hp.num_pass_xyz);
        lambda_stepsize_warmup  = η₁ * rand(float_type);
        omega_stepsize_warmup   = η₁ * rand(float_type);
        lambda_stepsize         = η₁ * rand(float_type, hp.num_pass_xyz);
        omega_stepsize          = η₁ * rand(float_type, hp.num_pass_xyz);

        new(
            lambda_sparsity_warmup, lambda_sparsity,
            lambda_stepsize_warmup, omega_stepsize_warmup, 
            lambda_stepsize, omega_stepsize,
            D, F
        )
    end
    function ucdl(lambda_sparity_warmup, 
                  lambda_sparsity, 
                  lambda_stepsize_warmup, 
                  omega_stepsize_warmup, 
                  lambda_stepsize, 
                  omega_stepsize, 
                  D, 
                  F
                )
        return new(
            lambda_sparity_warmup, 
            lambda_sparsity, 
            lambda_stepsize_warmup, 
            omega_stepsize_warmup, 
            lambda_stepsize, 
            omega_stepsize, 
            D, 
            F            
        )
    end
end

Flux.@functor ucdl


