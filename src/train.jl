function train(data; num_epochs=nothing)
    hp          = Hyperparam();
    len         = length_info(hp, data);
    projs       = projectors(hp, len);
    cdl         = ucdl(hp);

    data_load   = Flux.DataLoader(data.data_matrix, 
                                batchsize=hp.batch_size, 
                                shuffle=true, partial=false);
    ps          = Flux.params(cdl);
    opt         = Flux.AdaBelief();

    for i in 1:num_epochs
        for S in data_load
            S = S |> gpu;
            gs = gradient(ps) do
                forward_pass_return_loss(S, cdl, hp, len, projs)
            end
        end
    end
    return cdl, hp, len, projs
end