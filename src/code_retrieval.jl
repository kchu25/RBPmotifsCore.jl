
const float_type_retrieval = Float16

const stored_code_component_t = 
    NamedTuple{(:position, :fil, :seq, :mag), Tuple{UInt16, UInt16, UInt32, float_type_retrieval}}

get_code_component_this_batch(cartesian_ind, magnitude, i) = 
    (position=cartesian_ind[1], fil=cartesian_ind[3], seq=cartesian_ind[4]+i-1, mag=magnitude)

function append_code_component!(X, stored_code_components, i)
    cartesian_inds = findall(X .> 0);
    append!(stored_code_components, 
        get_code_component_this_batch.(cartesian_inds |> cpu, float_type_retrieval.(X[cartesian_inds]) |> cpu, i))
end

function code_retrieval(data, cdl, hp, len, projs)
    lambda_sparsity_warmup, lambda_sparsity,
    lambda_stepsize_warmup, omega_stepsize_warmup, lambda_stepsize, omega_stepsize, 
    D, F = RBPmotifsCore.prep_params(cdl, hp, projs)

    data_load = Flux.DataLoader(data.data_matrix, batchsize=hp.batch_size, shuffle=false, partial=false)
    stored_code_components = stored_code_component_t[]

    i = 1;
    for S in data_load
        S = S |> gpu;
        _, X = RBPmotifsCore.ADMM_XYZ(S, D, F,  
                    lambda_stepsize_warmup, lambda_stepsize, 
                    lambda_sparsity_warmup, lambda_sparsity,
                    omega_stepsize_warmup, omega_stepsize, 
                    hp, len, projs
                    )
        append_code_component!(X, stored_code_components, i)
        i += hp.batch_size;
    end

    return stored_code_components
end
