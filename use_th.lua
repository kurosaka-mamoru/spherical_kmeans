function spkmeans.use_th(x, k, th, batch_size, std)
    -- args
    x = x or error('missing argument: '.. help)
    k = k or error('missing argument: '.. help)
    batch_size = batch_size or 10000
    std = std or 0.1
    
    max_iter = 1000
    
    -- resize data and dims
    local nsamples = x:size(1)
    if x:dim() > 2 then
        x = x:reshape(nsamples, x:nElement() / nsamples)
    end
    local ndims = x:size(2)
    
    -- initialize centroids
    local centroids = torch.randn(k, ndims) * std
    
    -- normalize data and centroids
    local norms = torch.norm(x, 2, 2)
    x = torch.cdiv(x, torch.expand(norms, nsamples, ndims))
    local norms = torch.norm(centroids, 2, 2)
    centroids = torch.cdiv(centroids, torch.expand(norms, k, ndims))
    
    -- do iterations
    local labels = torch.Tensor(nsamples)
    for i = 1, max_iter do
        xlua.progress(i, niter)
        
        local old_val = 0 or val
        
        -- process batch
        local val = 0
        local labels = torch.Tensor()
        for i = 1, nsamples, batch_size do
            -- indices
            local lasti = math.min(i + batch_size - 1, nsamples)
            local m = lasti - i + 1
            
            -- update latent value
            local batch_t = x[{{i, lasti}, {}}]:t()
            local tmp = centroids * batch_t
            local val_tmp, labels_tmp = torch.max(tmp, 1)
            val = val + torch.sum(val_tmp)
            labels[{{i, lasti}}] = labels_tmp
        end
        
        -- update centroids
        local summation = torch.zeros(k, ndims)
        local S = torch.zeros(nsamples, k)
        for i = 1, labels:size(2) do
            S[i][labels[1][i]] = 1
        end
        summation = torch.add(summation, S:t() * x)
        local counts = torch.sum(S, 1):squeeze()
        centroids = summation
        
        -- update null cluster centroids
        for i = 1, k do
            if counts[i] == 0 then
                centroids[i] = torch.randn(k) * std
            end
        end
        
        -- normalize centroids
        local norms = torch.norm(centroids, 2, 2)
        centroids = torch.cdiv(centroids, torch.expand(norms, k, ndims))
        
        -- check termination condition
        if val - old_val < th then
            break
        end
    end
    
    -- done
    return centroids
end