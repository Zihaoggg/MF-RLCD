function write_orientation_files(microstructureId, nPriorBeta, oriBetaInput, nColoniesMin, nColoniesMax, outputDir, lamwidthBeta, lamwidthAlpha, lamwidthSingle, lamellarFraction, textureStrength)
    nPriorBeta = round(nPriorBeta);
    nColoniesMin = round(nColoniesMin);
    nColoniesMax = round(nColoniesMax);
    outputDir = char(outputDir);
    if ~(endsWith(outputDir, '/') || endsWith(outputDir, '\'))
        outputDir = [outputDir filesep];
    end

    nColonies = zeros(nPriorBeta, 1);
    betaEuler = zeros(nPriorBeta, 3);
    alphaEuler = zeros(nPriorBeta, nColoniesMax, 3);

    csBeta = crystalSymmetry('432', [3.32 3.32 3.32], 'mineral', 'Ti64 (beta)');
    csAlpha = crystalSymmetry('622', [2.95 2.95 4.68], 'mineral', 'Ti64 (alpha)');
    specimen = specimenSymmetry('1');

    if textureStrength == 0
        if is_random_orientation_input(oriBetaInput)
            oriBeta = project2FundamentalRegion(orientation.rand(nPriorBeta, csBeta));
        else
            betaInput = double(oriBetaInput);
            oriBeta = project2FundamentalRegion(orientation.byEuler(betaInput(:, 1) * degree, betaInput(:, 2) * degree, betaInput(:, 3) * degree, csBeta));
        end
    else
        textureCenter = orientation.byEuler(0, 0, 0, csBeta, specimen);
        odf = unimodalODF(textureCenter, 'halfwidth', textureStrength * degree);
        oriBeta = discreteSample(odf, nPriorBeta);
    end

    for betaId = 1:nPriorBeta
        betaEuler(betaId, :) = [oriBeta(betaId).phi1 / degree, oriBeta(betaId).Phi / degree, oriBeta(betaId).phi2 / degree];
    end

    betaToAlpha = orientation.Burgers(csBeta, csAlpha);
    oriBetaSym = oriBeta.symmetrise;
    oriBetaSym = oriBetaSym(1:12, :);

    for betaId = 1:nPriorBeta
        nColonies(betaId) = randi([nColoniesMin, nColoniesMax]);
        for colonyId = 1:nColonies(betaId)
            variantId = 1;
            colonyBeta = oriBetaSym(variantId, betaId);
            colonyAlpha = project2FundamentalRegion(colonyBeta * inv(betaToAlpha));
            alphaEuler(betaId, colonyId, :) = [colonyAlpha.phi1 / degree, colonyAlpha.Phi / degree, colonyAlpha.phi2 / degree];
        end
    end

    coloniesFile = fopen([outputDir sprintf('%g_colonies', microstructureId)], 'w');
    fprintf(coloniesFile, '%d %d\n', [1:nPriorBeta; nColonies']);
    fclose(coloniesFile);

    scale2File = fopen([outputDir sprintf('%g_scale2_ori', microstructureId)], 'w');
    lamwidthFile = fopen([outputDir sprintf('%g_lamwidth', microstructureId)], 'w');

    for betaId = 1:nPriorBeta
        fprintf(scale2File, '%d file(%g_cell%d,des=euler-bunge)\n', betaId, microstructureId, betaId);

        scale2CellFile = fopen([outputDir sprintf('%g_cell%d', microstructureId, betaId)], 'w');
        for colonyId = 1:nColoniesMax
            fprintf(scale2CellFile, '%f %f %f\n', squeeze(alphaEuler(betaId, colonyId, :)));
        end
        fclose(scale2CellFile);

        for colonyId = 1:nColonies(betaId)
            if rand() <= lamellarFraction
                fprintf(lamwidthFile, '%d::%d   %.2f:%.2f\n', betaId, colonyId, lamwidthAlpha, lamwidthBeta);
            else
                fprintf(lamwidthFile, '%d::%d   %.2f:%.2f\n', betaId, colonyId, lamwidthSingle, lamwidthSingle);
            end

            scale3CellFile = fopen([outputDir sprintf('%g_cell%d_%d', microstructureId, betaId, colonyId)], 'w');
            fprintf(scale3CellFile, '%f %f %f\n', squeeze(alphaEuler(betaId, colonyId, :)));
            fprintf(scale3CellFile, '%f %f %f\n', betaEuler(betaId, :));
            fclose(scale3CellFile);
        end
    end

    fclose(scale2File);
    fclose(lamwidthFile);
end

function flag = is_random_orientation_input(oriBetaInput)
    flag = isempty(oriBetaInput) || all(double(oriBetaInput(:)) == 0);
end
