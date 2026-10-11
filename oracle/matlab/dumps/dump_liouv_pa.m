% Oracle dump for sphten-liouv propagation: for each case, the Liouvil-
% lian with offsets and kinetics, the dwell-time propagator, step() on a
% vector and a stack, the trajectory-level reductions, evolution() in
% its three ported outputs, and the end-to-end pulse-acquire FID through
% liquid(). Syntax:
%
%                       dump_liouv_pa(out_root)
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

function dump_liouv_pa(out_root)

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

% Twelve-spin chain with nearest-neighbour couplings
chain.isotopes=repmat({'1H'},1,12); chain.magnet=14.1;
chain_inter.zeeman.scalar=num2cell(linspace(-4,4,12));
chain_inter.coupling.scalar=cell(12);
for n=1:11, chain_inter.coupling.scalar{n,n+1}=10; end

% Complete basis, strongly coupled proton pair
cases(1).name='ab';
cases(1).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(1).inter.zeeman.scalar={1.00 1.20};
cases(1).inter.coupling.scalar={0 12.0; 0 0};
cases(1).bas=struct('formalism','sphten-liouv','approximation','none');
cases(1).pars=struct('spins',{{'1H'}},'offset',660,'sweep',500,'npoints',256,'zerofill',1024);

% Complete basis with a spin-1
cases(2).name='hd';
cases(2).sys=struct('isotopes',{{'1H','2H'}},'magnet',14.1);
cases(2).inter.zeeman.scalar={4.50 4.50};
cases(2).inter.coupling.scalar={0 43.0; 0 0};
cases(2).bas=struct('formalism','sphten-liouv','approximation','none');
cases(2).pars=struct('spins',{{'1H'}},'offset',2700,'sweep',400,'npoints',256,'zerofill',1024);

% Three carbons, Krylov propagation forced through the tolerance
cases(3).name='inad_krylov';
cases(3).sys=struct('isotopes',{{'13C','13C','13C'}},'magnet',9.4);
cases(3).sys.tols.krylov_tol=1;
cases(3).inter.zeeman.scalar={10 15 20};
cases(3).inter.coupling.scalar=cell(3); cases(3).inter.coupling.scalar{1,2}=55;
cases(3).bas=struct('formalism','sphten-liouv','approximation','none');
cases(3).pars=struct('spins',{{'13C'}},'offset',1800,'sweep',5000,'npoints',256,'zerofill',1024);

% Heteronuclear pair with path tracing forced through the merge dimension
cases(4).name='hc_pt';
cases(4).sys=struct('isotopes',{{'1H','13C'}},'magnet',14.1);
cases(4).sys.tols.merge_dim=2;
cases(4).inter.zeeman.scalar={2.00 30.0};
cases(4).inter.coupling.scalar={0 145.0; 0 0};
cases(4).bas=struct('formalism','sphten-liouv','approximation','none');
cases(4).pars=struct('spins',{{'1H'}},'offset',1200,'sweep',500,'npoints',256,'zerofill',1024);

% Styrene, IK-0 at level 2 and IK-1 on the coupling graph
cases(5).name='styrene_ik0_2';
cases(5).sys=styrene; cases(5).inter=styrene_inter;
cases(5).bas=struct('formalism','sphten-liouv','approximation','IK-0','inter_level',2);
cases(5).pars=struct('spins',{{'1H'}},'offset',1200,'sweep',2000,'npoints',512,'zerofill',2048);
cases(6).name='styrene_ik1';
cases(6).sys=styrene; cases(6).inter=styrene_inter;
cases(6).bas=struct('formalism','sphten-liouv','approximation','IK-1',...
                    'connectivity','scalar_couplings','inter_level',3,'prox_level',1);
cases(6).pars=struct('spins',{{'1H'}},'offset',1200,'sweep',2000,'npoints',512,'zerofill',2048);

% Twelve-spin chain, IK-2
cases(7).name='chain12_ik2';
cases(7).sys=chain; cases(7).inter=chain_inter;
cases(7).bas=struct('formalism','sphten-liouv','approximation','IK-2',...
                    'connectivity','scalar_couplings','prox_level',1);
cases(7).pars=struct('spins',{{'1H'}},'offset',0,'sweep',8000,'npoints',512,'zerofill',2048);

% Symmetric and asymmetric two-site exchange (examples/kinetics/exchange_*.m)
cases(8).name='exch_sym';
cases(8).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(8).inter.zeeman.scalar={0.0 3.0};
cases(8).inter.chem.parts={1,2};
cases(8).inter.chem.rates=[-2e3 2e3; 2e3 -2e3];
cases(8).inter.chem.concs=[1.0 1.0];
cases(8).bas=struct('formalism','sphten-liouv','approximation','none');
cases(8).pars=struct('spins',{{'1H'}},'offset',900,'sweep',5000,'npoints',512,'zerofill',1024);
cases(9).name='exch_asym';
cases(9).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(9).inter.zeeman.scalar={0.0 3.0};
cases(9).inter.chem.parts={1,2};
cases(9).inter.chem.rates=[-5e2 2e3; 5e2 -2e3];
cases(9).inter.chem.concs=[2e3 5e2];
cases(9).bas=struct('formalism','sphten-liouv','approximation','none');
cases(9).pars=struct('spins',{{'1H'}},'offset',900,'sweep',5000,'npoints',512,'zerofill',1024);

% Settings common to all cases
dump_file=[mfilename('fullpath') '.m'];
ntraj=32;      % trajectory points stored per case

% Loop over cases
for n=1:numel(cases)

    % Never overwrite existing fixtures: skip cases that are already there
    case_dir=fullfile(out_root,['liouv_pa_' cases(n).name]);
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
    inter=cases(n).inter; bas=cases(n).bas; pars=cases(n).pars; det=pars.spins{1};
    spin_system=create(sys,inter);
    spin_system=basis(spin_system,bas);
    dt=1/pars.sweep;
    oracle_write(h5_file,'/bas/basis',spin_system.bas.basis);

    % Liouvillian with offsets and kinetics, as liquid() and acquire() form it
    progress(case_label,'Liouvillian');
    nmr=assume(spin_system,'nmr');
    H=frqoffset(spin_system,hamiltonian(nmr),pars);
    K=kinetics(spin_system);
    L=H+1i*K;
    oracle_write(h5_file,'/ham/L',L);

    % Initial and detection states
    rho0=state(spin_system,'L+',det,'chem');
    coil=state(spin_system,'L+',det);
    oracle_write(h5_file,'/states/rho0',rho0);
    oracle_write(h5_file,'/states/coil',coil);

    % Propagator over one dwell time
    progress(case_label,'propagator');
    oracle_write(h5_file,'/prop/P',propagator(spin_system,L,dt));

    % Taylor steps: one dwell time, a long step needing substeps, and a stack
    progress(case_label,'step');
    oracle_write(h5_file,'/step/rho_dt',step(spin_system,L,rho0,dt));
    oracle_write(h5_file,'/step/rho_long',step(spin_system,L,rho0,7.3*dt));
    oracle_write(h5_file,'/step/stack',step(spin_system,L,[rho0 0.5*coil],dt));

    % Trajectory-level reductions as evolution() applies them for the observable output
    progress(case_label,'reductions');
    oracle_write(h5_file,'/reduce/zte_projector',zte(spin_system,L',coil));
    oracle_write(h5_file,'/reduce/projectors',reduce(spin_system,L',coil));

    % Kernel evolution function, three outputs
    progress(case_label,'evolution');
    oracle_write(h5_file,'/fid/evolution',evolution(spin_system,L,coil,rho0,dt,pars.npoints-1,'observable'));
    oracle_write(h5_file,'/fid/final',evolution(spin_system,L,[],rho0,dt,pars.npoints-1,'final'));
    oracle_write(h5_file,'/fid/trajectory',evolution(spin_system,L,[],rho0,dt,ntraj-1,'trajectory'));

    % End-to-end context and sequence
    progress(case_label,'liquid() and acquire()');
    seq_pars=pars; seq_pars.rho0=rho0; seq_pars.coil=coil; seq_pars.decouple={};
    fid=liquid(spin_system,@acquire,seq_pars,'nmr');
    oracle_write(h5_file,'/fid/liquid',fid);
    oracle_write(h5_file,'/spectrum/raw',fftshift(fft(fid,pars.zerofill)));

    % Provenance record
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'pars',pars,'ntraj',ntraj,...
                  'rho0','L+ on detected isotope, chem','coil','L+ on detected isotope, exact',...
                  'assumptions','nmr','timestep',dt);
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
