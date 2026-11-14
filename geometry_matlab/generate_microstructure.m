function generate_microstructure(nPriorBeta, oriBetaInput, nColoniesMax, lamwidthBeta, lamRatio, microstructureId, runName, orientationId, outputDir)
    rng('default');
    rng(microstructureId);
    outputDir = char(outputDir);
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end
    write_neper_script(nPriorBeta, oriBetaInput, nColoniesMax, nColoniesMax, ...
        lamwidthBeta, outputDir, microstructureId, lamRatio, 0);
end
