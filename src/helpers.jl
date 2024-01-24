
function get_data_bg(data)
    this_bg = reshape(sum(reshape(data.data_matrix, (4, data.L, data.N)), dims=(2,3)), (4,))
    this_bg = float_type.(this_bg ./ sum(this_bg))
    return this_bg
end