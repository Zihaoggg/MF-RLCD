function write_neper_script(nPriorBeta, oriBetaInput, nColoniesMin, nColoniesMax, lamwidthBeta, outputDir, microstructureId, lamRatio, textureStrength)
    nPriorBeta = round(nPriorBeta);
    nColoniesMin = round(nColoniesMin);
    nColoniesMax = round(nColoniesMax);
    outputDir = char(outputDir);
    if ~(endsWith(outputDir, '/') || endsWith(outputDir, '\'))
        outputDir = [outputDir filesep];
    end

    domainLength = 1;
    domainWidth = 1;
    domainThickness = 1;
    diameqMean = 1;
    diameqStd = 0.2;
    sphericityMean = 0.855;
    sphericityStd = 0.03;
    lamwidthAlpha = lamwidthBeta * lamRatio;
    lamwidthSingle = domainLength * 10;
    lamellarFraction = 1.0;

    morpho1 = sprintf('diameq:lognormal(%0.4f,%0.4f),sphericity:normal(%0.4f,%0.4f)', diameqMean, diameqStd, sphericityMean, sphericityStd);
    morpho2 = sprintf('diameq:lognormal(%0.4f,%0.4f),sphericity:normal(%0.4f,%0.4f)', diameqMean, diameqStd, sphericityMean, sphericityStd);
    morpho3 = sprintf('lamellar(w=file(%g_lamwidth),v=crysdir(-1,1,0),pos=optimal)', microstructureId);
    morpho = sprintf('%s::%s::%s', morpho1, morpho2, morpho3);
    nGrains = sprintf('%d::file(%g_colonies)::from_morpho', nPriorBeta, microstructureId);
    outputStem = sprintf('%d_%d_%d_%d_%g', nPriorBeta, nColoniesMax, round(lamwidthBeta * 100000), round(lamRatio * 100000), microstructureId);
    selection = sprintf('%0.10g', lamwidthBeta * 0.1);
    useSeedFile = ~is_random_orientation_input(oriBetaInput);

    scriptPath = [outputDir 'generate_tess.sh'];
    script = fopen(scriptPath, 'w');
    fprintf(script, '#!/bin/bash\n');
    fprintf(script, 'cd %s\n', shell_quote(outputDir));

    write_orientation_files( ...
        microstructureId, ...
        nPriorBeta, ...
        oriBetaInput, ...
        nColoniesMin, ...
        nColoniesMax, ...
        outputDir, ...
        lamwidthBeta, ...
        lamwidthAlpha, ...
        lamwidthSingle, ...
        lamellarFraction, ...
        textureStrength ...
    );

    if useSeedFile
        seedArg = '-morphooptiini ''coo:file(seeds)'' ';
    else
        seedArg = '';
    end

    fprintf(script, "neper -T -n '%s' -id %g %s-dim 3 -domain 'cube(%g,%g,%g)' -ori 'random::file(%g_scale2_ori)::random' -morpho '%s' -group 'lam==1?1:2' -statcell ""scaleid(1),lam"" -statgroup vol -reg 1 -sel %s -format tess -o %s\n", ...
        nGrains, microstructureId, seedArg, domainLength, domainWidth, domainThickness, microstructureId, morpho, selection, outputStem);

    fprintf(script, 'rm -f %g_scale3_ori\n', microstructureId);
    for betaId = 1:nPriorBeta
        for colonyId = 1:nColoniesMax
            fprintf(script, "awk '{if ($1==%d) print $2}' %s.stcell > %g_ori-grain%d\n", betaId, outputStem, microstructureId, betaId);
            fprintf(script, "ORI_A=$(awk 'NR==1 {print $1,$2,$3}' %g_cell%d_%d)\n", microstructureId, betaId, colonyId);
            fprintf(script, "ORI_B=$(awk 'NR==2 {print $1,$2,$3}' %g_cell%d_%d)\n", microstructureId, betaId, colonyId);
            fprintf(script, "sed -i ""s/^\\<1\\>$/$ORI_A/"" %g_ori-grain%d\n", microstructureId, betaId);
            fprintf(script, "sed -i ""s/^\\<2\\>$/$ORI_B/"" %g_ori-grain%d\n", microstructureId, betaId);
            fprintf(script, 'echo "%d::%d file(%g_ori-grain%d,des=euler-bunge)" >> %g_scale3_ori\n', betaId, colonyId, microstructureId, betaId, microstructureId);
        end
    end

    fprintf(script, "neper -T -n '%s' -id %g %s-dim 3 -domain 'cube(%g,%g,%g)' -ori 'random::file(%g_scale2_ori)::file(%g_scale3_ori)' -morpho '%s' -group 'lam==1?1:2' -statgroup vol -reg 1 -sel %s -format tess -o %s\n", ...
        nGrains, microstructureId, seedArg, domainLength, domainWidth, domainThickness, microstructureId, microstructureId, morpho, selection, outputStem);
    fprintf(script, "neper -T -n '%s' -id %g %s-dim 3 -domain 'cube(%g,%g,%g)' -ori 'random::file(%g_scale2_ori)::file(%g_scale3_ori)' -morpho '%s' -reg 1 -sel %s -format tesr -tesrformat 'ascii' -tesrsize 32 -o %s\n", ...
        nGrains, microstructureId, seedArg, domainLength, domainWidth, domainThickness, microstructureId, microstructureId, morpho, selection, outputStem);

    fprintf(script, "neper -V %s.tess -datacellcol 'ori' -datacellcolscheme 'ipf(z)' -print %s_geom\n", outputStem, outputStem);
    fprintf(script, "neper -V %s.tesr -datacellcol 'ori' -datacellcolscheme 'ipf(z)' -print %s_geom_raster\n", outputStem, outputStem);
    fprintf(script, 'exit 0\n');
    fclose(script);
end

function flag = is_random_orientation_input(oriBetaInput)
    flag = isempty(oriBetaInput) || all(double(oriBetaInput(:)) == 0);
end

function text = shell_quote(pathText)
    quote = char(39);
    escaped = strrep(pathText, quote, [quote '"' quote '"' quote]);
    text = [quote escaped quote];
end
