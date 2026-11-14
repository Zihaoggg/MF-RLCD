function write_neper_script(nPriorBeta, oriBetaInput, nColoniesMin, nColoniesMax, lamwidthBeta, outputDir, microstructureId, lamRatio, textureStrength)
    nPriorBeta = round(nPriorBeta);
    nColoniesMax = round(nColoniesMax);
    outputDir = char(outputDir);
    lamwidthAlpha = lamwidthBeta * lamRatio;
    write_orientation_files(microstructureId, nPriorBeta, oriBetaInput, ...
        nColoniesMin, nColoniesMax, outputDir, lamwidthBeta, lamwidthAlpha, 10, 1, textureStrength);

    stem = sprintf('%d_%d_%d_%d_%g', nPriorBeta, nColoniesMax, ...
        round(lamwidthBeta * 100000), round(lamRatio * 100000), microstructureId);
    grainCount = sprintf('%d::file(%g_colonies)::from_morpho', nPriorBeta, microstructureId);
    baseMorphology = 'diameq:lognormal(1.0000,0.2000),sphericity:normal(0.8550,0.0300)';
    morphology = sprintf('%s::%s::lamellar(w=file(%g_lamwidth),v=crysdir(-1,1,0),pos=optimal)', ...
        baseMorphology, baseMorphology, microstructureId);
    seedOption = '';
    if ~isempty(oriBetaInput) && any(double(oriBetaInput(:)) ~= 0)
        seedOption = '-morphooptiini ''coo:file(seeds)'' ';
    end
    baseCommand = sprintf('neper -T -n ''%s'' -id %g %s-dim 3 -domain ''cube(1,1,1)'' ', ...
        grainCount, microstructureId, seedOption);
    options = sprintf('-morpho ''%s'' -reg 1 -sel %g ', morphology, lamwidthBeta * 0.1);

    script = fopen(fullfile(outputDir, 'generate_tess.sh'), 'w');
    if script < 0
        error('Cannot write Neper shell script in %s', outputDir);
    end
    closeScript = onCleanup(@() fclose(script));
    fprintf(script, '#!/bin/bash\n');
    fprintf(script, 'cd %s\n', shell_quote(outputDir));
    fprintf(script, '%s-ori ''random::file(%g_scale2_ori)::random'' %s-group ''lam==1?1:2'' -statcell "scaleid(1),lam" -statgroup vol -format tess -o %s\n', ...
        baseCommand, microstructureId, options, stem);
    fprintf(script, 'rm -f %g_scale3_ori\n', microstructureId);
    for betaId = 1:nPriorBeta
        for colonyId = 1:nColoniesMax
            fprintf(script, 'awk ''{if ($1==%d) print $2}'' %s.stcell > %g_ori-grain%d\n', betaId, stem, microstructureId, betaId);
            fprintf(script, 'ORI_A=$(awk ''NR==1 {print $1,$2,$3}'' %g_cell%d_%d)\n', microstructureId, betaId, colonyId);
            fprintf(script, 'ORI_B=$(awk ''NR==2 {print $1,$2,$3}'' %g_cell%d_%d)\n', microstructureId, betaId, colonyId);
            fprintf(script, 'sed -i "s/^\\<1\\>$/$ORI_A/" %g_ori-grain%d\n', microstructureId, betaId);
            fprintf(script, 'sed -i "s/^\\<2\\>$/$ORI_B/" %g_ori-grain%d\n', microstructureId, betaId);
            fprintf(script, 'echo "%d::%d file(%g_ori-grain%d,des=euler-bunge)" >> %g_scale3_ori\n', ...
                betaId, colonyId, microstructureId, betaId, microstructureId);
        end
    end
    orientations = sprintf('-ori ''random::file(%g_scale2_ori)::file(%g_scale3_ori)'' ', microstructureId, microstructureId);
    fprintf(script, '%s%s%s-group ''lam==1?1:2'' -statgroup vol -format tess -o %s\n', baseCommand, orientations, options, stem);
    fprintf(script, '%s%s%s-format tesr -tesrformat ''ascii'' -tesrsize 32 -o %s\n', baseCommand, orientations, options, stem);
    fprintf(script, 'neper -V %s.tess -datacellcol ''ori'' -datacellcolscheme ''ipf(z)'' -print %s_geom\n', stem, stem);
    fprintf(script, 'neper -V %s.tesr -datacellcol ''ori'' -datacellcolscheme ''ipf(z)'' -print %s_geom_raster\n', stem, stem);
    fprintf(script, 'exit 0\n');
end

function quoted = shell_quote(text)
    quote = char(39);
    quoted = [quote strrep(text, quote, [quote '"' quote '"' quote]) quote];
end
