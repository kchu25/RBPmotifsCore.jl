function train(data; num_epochs=5)
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
        @info "Epoch $i"
        for S in data_load
            S = S |> gpu;
            gs = gradient(ps) do
                forward_pass_return_loss(S, cdl, hp, len, projs)
            end
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
    end
    return cdl, hp, len, projs
end
