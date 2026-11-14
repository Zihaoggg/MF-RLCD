function write_orientation_files(microstructureId, nPriorBeta, oriBetaInput, nColoniesMin, nColoniesMax, outputDir, lamwidthBeta, lamwidthAlpha, lamwidthSingle, lamellarFraction, textureStrength)
    nPriorBeta = round(nPriorBeta);
    nColoniesMin = round(nColoniesMin);
    nColoniesMax = round(nColoniesMax);
    outputDir = char(outputDir);
    csBeta = crystalSymmetry('432', [3.32 3.32 3.32], 'mineral', 'Ti64 (beta)');
    csAlpha = crystalSymmetry('622', [2.95 2.95 4.68], 'mineral', 'Ti64 (alpha)');

    if textureStrength == 0
        if isempty(oriBetaInput) || all(double(oriBetaInput(:)) == 0)
            oriBeta = project2FundamentalRegion(orientation.rand(nPriorBeta, csBeta));
        else
            angles = double(oriBetaInput);
            if ~isequal(size(angles), [nPriorBeta 3])
                error('Expected one row of three Euler angles per beta grain.');
            end
            oriBeta = project2FundamentalRegion(orientation.byEuler( ...
                angles(:, 1) * degree, angles(:, 2) * degree, angles(:, 3) * degree, csBeta));
        end
    else
        center = orientation.byEuler(0, 0, 0, csBeta, specimenSymmetry('1'));
        odf = unimodalODF(center, 'halfwidth', textureStrength * degree);
        oriBeta = discreteSample(odf, nPriorBeta);
    end

    betaToAlpha = orientation.Burgers(csBeta, csAlpha);
    symmetricBeta = oriBeta.symmetrise;
    symmetricBeta = symmetricBeta(1:12, :);
    colonies = zeros(nPriorBeta, 1);
    betaAngles = zeros(nPriorBeta, 3);
    alphaAngles = zeros(nPriorBeta, nColoniesMax, 3);
    for betaId = 1:nPriorBeta
        betaAngles(betaId, :) = [oriBeta(betaId).phi1, oriBeta(betaId).Phi, oriBeta(betaId).phi2] / degree;
        colonies(betaId) = randi([nColoniesMin, nColoniesMax]);
        for colonyId = 1:colonies(betaId)
            child = project2FundamentalRegion(symmetricBeta(1, betaId) * inv(betaToAlpha));
            alphaAngles(betaId, colonyId, :) = [child.phi1, child.Phi, child.phi2] / degree;
        end
    end

    file = open_output(outputDir, sprintf('%g_colonies', microstructureId));
    fprintf(file, '%d %d\n', [1:nPriorBeta; colonies']);
    fclose(file);
    scale2 = open_output(outputDir, sprintf('%g_scale2_ori', microstructureId));
    closeScale2 = onCleanup(@() fclose(scale2));
    widths = open_output(outputDir, sprintf('%g_lamwidth', microstructureId));
    closeWidths = onCleanup(@() fclose(widths));
    for betaId = 1:nPriorBeta
        fprintf(scale2, '%d file(%g_cell%d,des=euler-bunge)\n', betaId, microstructureId, betaId);
        file = open_output(outputDir, sprintf('%g_cell%d', microstructureId, betaId));
        for colonyId = 1:nColoniesMax
            fprintf(file, '%f %f %f\n', reshape(alphaAngles(betaId, colonyId, :), 1, 3));
        end
        fclose(file);
        for colonyId = 1:colonies(betaId)
            if rand() <= lamellarFraction
                fprintf(widths, '%d::%d   %.2f:%.2f\n', betaId, colonyId, lamwidthAlpha, lamwidthBeta);
            else
                fprintf(widths, '%d::%d   %.2f:%.2f\n', betaId, colonyId, lamwidthSingle, lamwidthSingle);
            end
            file = open_output(outputDir, sprintf('%g_cell%d_%d', microstructureId, betaId, colonyId));
            fprintf(file, '%f %f %f\n', reshape(alphaAngles(betaId, colonyId, :), 1, 3));
            fprintf(file, '%f %f %f\n', betaAngles(betaId, :));
            fclose(file);
        end
    end
end

function file = open_output(directory, name)
    file = fopen(fullfile(directory, name), 'w');
    if file < 0
        error('Cannot write orientation file: %s', fullfile(directory, name));
    end
end
