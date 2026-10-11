% Oracle dump for the sphten-liouv setup slice: basis descriptor, pro-
% duct tables, superoperators of the four types, states by the three
% methods, the Hamiltonian commutation superoperator and its parts, the
% kinetics superoperator, and the analytical filters. Nothing is propa-
% gated, so every case takes seconds. Syntax:
%
%                       dump_liouv_setup(out_root)
%
% Parameters:
%
%    out_root - fixture directory for the current Spinach tag, e.g.
%               /fixtures/<spinach-tag>; cases whose directory
%               already exists there are skipped, never overwritten
%
% Outputs:
%
%    this function writes fixture files, see FIXTURE_REQUESTS.md and
%    oracle/README.md for the list of variables
%
% Note: oracle/matlab/lib must be on the Matlab path, and Spinach must
%       be on the path with no modifications. Each case also writes
%       /probe variables with known content that verify the HDF5 lay-
%       out conventions on the Python side.

function dump_liouv_setup(out_root)

% Check consistency
grumble(out_root);

% Styrene, 8 protons at 7.046 T (examples/nmr_liquids/pa_styrene.m)
styrene.isotopes={'1H','1H','1H','1H','1H','1H','1H','1H'};
styrene.magnet=7.046;
styrene_inter.zeeman.scalar={6.720  6.720  6.587  6.587  6.527  6.063   5.085  4.555};
styrene_inter.coupling.scalar={ 0.0       1.9170    7.7884    0.6061    1.2447   -0.5347    0.0390    0.1582
                                0.0       0.0       0.6061    7.7884    1.2447   -0.5347    0.0390    0.1582
                                0.0       0.0       0.0       1.4184    7.4390    0.3693   -0.0416    0.0169
                                0.0       0.0       0.0       0.0       7.4390    0.3693   -0.0416    0.0169
                                0.0       0.0       0.0       0.0       0.0      -0.2279    0.2349    0.2905
                                0.0       0.0       0.0       0.0       0.0       0.0      17.6002   10.8954
                                0.0       0.0       0.0       0.0       0.0       0.0       0.0       1.0382
                                0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0};

% Twelve-spin chain with nearest-neighbour couplings (shortened
% examples/fundamentals/quadratures/product_quadratures_1.m)
chain.isotopes=repmat({'1H'},1,12); chain.magnet=14.1;
chain_inter.zeeman.scalar=num2cell(linspace(-4,4,12));
chain_inter.coupling.scalar=cell(12);
for n=1:11, chain_inter.coupling.scalar{n,n+1}=10; end

% Complete basis, strongly coupled proton pair (FR-001 ab_2spin)
cases(1).name='ab_none';
cases(1).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(1).inter.zeeman.scalar={1.00 1.20};
cases(1).inter.coupling.scalar={0 12.0; 0 0};
cases(1).bas=struct('formalism','sphten-liouv','approximation','none');
cases(1).det='1H'; cases(1).offset=660; cases(1).cheap={'Lz','L+'};

% Complete basis with a spin-1 (FR-001 hd_2spin)
cases(2).name='hd_none';
cases(2).sys=struct('isotopes',{{'1H','2H'}},'magnet',14.1);
cases(2).inter.zeeman.scalar={4.50 4.50};
cases(2).inter.coupling.scalar={0 43.0; 0 0};
cases(2).bas=struct('formalism','sphten-liouv','approximation','none');
cases(2).det='1H'; cases(2).offset=2700; cases(2).cheap={'Lz','L+'};

% Three carbons (examples/nmr_liquids/inad_three_spin.m)
cases(3).name='inad_none';
cases(3).sys=struct('isotopes',{{'13C','13C','13C'}},'magnet',9.4);
cases(3).inter.zeeman.scalar={10 15 20};
cases(3).inter.coupling.scalar=cell(3); cases(3).inter.coupling.scalar{1,2}=55;
cases(3).bas=struct('formalism','sphten-liouv','approximation','none');
cases(3).det='13C'; cases(3).offset=1800; cases(3).cheap={'Lz','L+'};

% Styrene, IK-0 at three correlation levels
levels=[1 2 4];
for k=1:3
    cases(3+k).name=['styrene_ik0_' int2str(levels(k))];
    cases(3+k).sys=styrene; cases(3+k).inter=styrene_inter;
    cases(3+k).bas=struct('formalism','sphten-liouv','approximation','IK-0','inter_level',levels(k));
    cases(3+k).det='1H'; cases(3+k).offset=1200; cases(3+k).cheap={'Lz','L+'};
end

% Styrene, IK-1 on the coupling graph
cases(7).name='styrene_ik1';
cases(7).sys=styrene; cases(7).inter=styrene_inter;
cases(7).bas=struct('formalism','sphten-liouv','approximation','IK-1',...
                    'connectivity','scalar_couplings','inter_level',3,'prox_level',1);
cases(7).det='1H'; cases(7).offset=1200; cases(7).cheap={'Lz','L+'};

% Styrene, IK-2 with a coherence order filter
cases(8).name='styrene_ik2_proj';
cases(8).sys=styrene; cases(8).inter=styrene_inter;
cases(8).bas=struct('formalism','sphten-liouv','approximation','IK-2',...
                    'connectivity','scalar_couplings','prox_level',1,'projections',{{1}});
cases(8).det='1H'; cases(8).offset=1200; cases(8).cheap={'L+'};

% Twelve-spin chain, IK-2
cases(9).name='chain12_ik2';
cases(9).sys=chain; cases(9).inter=chain_inter;
cases(9).bas=struct('formalism','sphten-liouv','approximation','IK-2',...
                    'connectivity','scalar_couplings','prox_level',1);
cases(9).det='1H'; cases(9).offset=0; cases(9).cheap={'Lz','L+'};

% Symmetric two-site exchange (examples/kinetics/exchange_symmetric.m)
cases(10).name='exch_sym';
cases(10).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(10).inter.zeeman.scalar={0.0 3.0};
cases(10).inter.chem.parts={1,2};
cases(10).inter.chem.rates=[-2e3 2e3; 2e3 -2e3];
cases(10).inter.chem.concs=[1.0 1.0];
cases(10).bas=struct('formalism','sphten-liouv','approximation','none');
cases(10).det='1H'; cases(10).offset=900; cases(10).cheap={'Lz','L+'};

% Asymmetric two-site exchange (examples/kinetics/exchange_asymmetric.m)
cases(11).name='exch_asym';
cases(11).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(11).inter.zeeman.scalar={0.0 3.0};
cases(11).inter.chem.parts={1,2};
cases(11).inter.chem.rates=[-5e2 2e3; 5e2 -2e3];
cases(11).inter.chem.concs=[2e3 5e2];
cases(11).bas=struct('formalism','sphten-liouv','approximation','none');
cases(11).det='1H'; cases(11).offset=900; cases(11).cheap={'Lz','L+'};

% 1H-13C-15N with longitudinal and zero-quantum filters
cases(12).name='hcn_filters';
cases(12).sys=struct('isotopes',{{'1H','13C','15N'}},'magnet',14.1);
cases(12).inter.zeeman.scalar={4.70 55.0 120.0};
cases(12).inter.coupling.scalar={0 140.0 -92.0; 0 0 -15.0; 0 0 0};
cases(12).bas=struct('formalism','sphten-liouv','approximation','none',...
                     'longitudinal',{{{'15N'}}},'zero_quantum',{{{[1 2]}}});
cases(12).det='13C'; cases(12).offset=500; cases(12).cheap={'Lz'};

% Settings common to all cases
dump_file=[mfilename('fullpath') '.m'];
op_names={'E','Lz','Lx','Ly','L+','L-'};
op_keys={'E','Lz','Lx','Ly','Lp','Lm'};
op_types={'left','right','comm','acomm'};

% Loop over cases
for n=1:numel(cases)

    % Never overwrite existing fixtures: skip cases that are already there
    case_dir=fullfile(out_root,['liouv_' cases(n).name]);
    if isfolder(case_dir)
        fprintf('%s [%s] exists, skipped\n',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')),cases(n).name);
        continue;
    end
    mkdir(case_dir); h5_file=fullfile(case_dir,'data.h5');
    case_label=sprintf('%s %d/%d',cases(n).name,n,numel(cases));
    progress(case_label,['started, writing to ' case_dir]);

    % Layout convention probes with known content
    oracle_write(h5_file,'/probe/dense',reshape(1:24,[2 3 4])+1i*reshape(101:124,[2 3 4]));
    oracle_write(h5_file,'/probe/column',[1; 2; 3]);
    oracle_write(h5_file,'/probe/sparse',sparse([1 3 2],[2 1 4],[5+1i 7 -2i],3,4));
    oracle_write(h5_file,'/probe/cell',{[1 2],sparse(eye(2)); [],7});

    % Spin system and basis
    progress(case_label,'create and basis');
    sys=cases(n).sys; sys.output=fullfile(case_dir,'spinach.log');
    sys.disable={'hygiene'}; % existentials() would reject the isworkernode.m shadow
    inter=cases(n).inter; bas=cases(n).bas; det=cases(n).det;
    spin_system=create(sys,inter);
    spin_system=basis(spin_system,bas);
    nspins=spin_system.comp.nspins;
    oracle_write(h5_file,'/comp/mults',spin_system.comp.mults);
    oracle_write(h5_file,'/bas/basis',spin_system.bas.basis);
    oracle_write(h5_file,'/bas/tot_proj',spin_system.bas.tot_proj);
    oracle_write(h5_file,'/bas/tot_cord',spin_system.bas.tot_cord);
    for mult=unique(spin_system.comp.mults(:))'
        oracle_write(h5_file,['/bas/lpst_' int2str(mult)],spin_system.bas.lpst{mult});
        oracle_write(h5_file,['/bas/rpst_' int2str(mult)],spin_system.bas.rpst{mult});
    end
    progress(case_label,[int2str(size(spin_system.bas.basis,1)) ' states']);

    % Single-spin superoperators of every type
    progress(case_label,'operators');
    for t=1:numel(op_types)
        for k=1:numel(op_names)
            single_ops=cell(1,nspins);
            for m=1:nspins
                single_ops{m}=operator(spin_system,op_names(k),{m},op_types{t});
            end
            oracle_write(h5_file,['/ops/' op_keys{k} '_' op_types{t}],single_ops);
        end
        if nspins>1
            oracle_write(h5_file,['/ops/LzLp_12_' op_types{t}],operator(spin_system,{'Lz','L+'},{1,2},op_types{t}));
        end
    end
    oracle_write(h5_file,'/ops/Lz_det_comm',operator(spin_system,'Lz',det));
    oracle_write(h5_file,'/ops/Lp_det_left',operator(spin_system,'L+',det,'left'));

    % States by the three methods on the detected isotope
    progress(case_label,'states');
    oracle_write(h5_file,'/states/Lz_exact',state(spin_system,'Lz',det));
    oracle_write(h5_file,'/states/Lp_exact',state(spin_system,'L+',det));
    oracle_write(h5_file,'/states/Lz_chem',state(spin_system,'Lz',det,'chem'));
    oracle_write(h5_file,'/states/Lp_chem',state(spin_system,'L+',det,'chem'));
    if ismember('Lz',cases(n).cheap)
        oracle_write(h5_file,'/states/Lz_cheap',state(spin_system,'Lz',det,'cheap'));
    end
    if ismember('L+',cases(n).cheap)
        oracle_write(h5_file,'/states/Lp_cheap',state(spin_system,'L+',det,'cheap'));
    end
    if nspins>1
        oracle_write(h5_file,'/states/LzLp_12_exact',state(spin_system,{'Lz','L+'},{1,2}));
    end

    % Hamiltonian superoperators
    progress(case_label,'Hamiltonian');
    nmr=assume(spin_system,'nmr');
    H=hamiltonian(nmr);
    oracle_write(h5_file,'/ham/H',H);
    oracle_write(h5_file,'/ham/H_zeeman',hamiltonian(assume(spin_system,'nmr','zeeman')));
    oracle_write(h5_file,'/ham/H_coupling',hamiltonian(assume(spin_system,'nmr','couplings')));
    oracle_write(h5_file,'/ham/H_left',hamiltonian(nmr,'left'));
    oracle_write(h5_file,'/ham/H_right',hamiltonian(nmr,'right'));
    pars=struct('spins',{{det}},'offset',cases(n).offset);
    oracle_write(h5_file,'/ham/H_off',frqoffset(spin_system,H,pars));

    % Kinetics superoperator
    progress(case_label,'kinetics');
    oracle_write(h5_file,'/kin/K',kinetics(spin_system));

    % Analytical filters on a dense test vector
    progress(case_label,'filters');
    nstates=size(spin_system.bas.basis,1);
    v=(1:nstates)'+1i*(nstates:-1:1)';
    oracle_write(h5_file,'/filters/input',v);
    oracle_write(h5_file,'/filters/corr_12',correlation(spin_system,v,[1 2]));
    oracle_write(h5_file,'/filters/corr_1_det',correlation(spin_system,v,1,det));
    oracle_write(h5_file,'/filters/coh_det_p1',coherence(spin_system,v,{{det,1}}));
    oracle_write(h5_file,'/filters/coh_all_0',coherence(spin_system,v,{{'all',0}}));

    % Provenance record
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'det',det,...
                  'offset',cases(n).offset,'cheap',{cases(n).cheap},'assumptions','nmr');
    oracle_meta(case_dir,dump_file,spin_system,inputs);

    % Close the Spinach log
    fclose(spin_system.sys.output);
    progress(case_label,'finished');

end

end

% Timestamped progress line on the console
function progress(case_label,stage)
fprintf('%s [%s] %s\n',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')),case_label,stage);
end

% Consistency enforcement
function grumble(out_root)
if (~ischar(out_root))||(~isfolder(out_root))
    error('out_root must be the name of an existing directory.');
end
end
