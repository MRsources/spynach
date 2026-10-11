% Oracle dump for relaxation: Redfield, extended T1/T2 and damping super-
% operators with the retention and thermalisation policies, the thermal
% equilibrium states, and transient NOE trajectories through the multi-
% channel evolution output. Syntax:
%
%                       dump_relax(out_root)
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

function dump_relax(out_root)

% Check consistency
grumble(out_root);

% Heteronuclear NOE pair (examples/nmr_liquids/noe_two_spin_het.m)
cases(1).name='noe_het';
cases(1).sys=struct('isotopes',{{'1H','13C'}},'magnet',14.1);
cases(1).inter.zeeman.scalar={0.0 0.0};
cases(1).inter.coordinates={[0 0 0]; [0 0 1.03]};
cases(1).inter.relaxation={'redfield'};
cases(1).inter.equilibrium='dibari';
cases(1).inter.rlx_keep='kite';
cases(1).inter.temperature=298;
cases(1).inter.tau_c={100e-12};
cases(1).bas=struct('formalism','sphten-liouv','approximation','none');
cases(1).noe=struct('dt',1e-2,'nsteps',400);

% Homonuclear NOE pair (examples/nmr_liquids/noe_two_spin_hom.m)
cases(2).name='noe_hom';
cases(2).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(2).inter.zeeman.scalar={0.0 0.0};
cases(2).inter.coordinates={[0 0 0]; [0 0 2.00]};
cases(2).inter.relaxation={'redfield'};
cases(2).inter.equilibrium='dibari';
cases(2).inter.rlx_keep='kite';
cases(2).inter.temperature=298;
cases(2).inter.tau_c={1e-9};
cases(2).bas=struct('formalism','sphten-liouv','approximation','none');
cases(2).noe=struct('dt',1e-2,'nsteps',1000);

% Three spins with CSA and dipolar couplings, full Redfield with secular retention and IME
cases(3).name='csa_dd_secular';
cases(3).sys=struct('isotopes',{{'1H','1H','13C'}},'magnet',14.1);
cases(3).inter.zeeman.scalar={1.0 2.5 50.0};
cases(3).inter.zeeman.eigs={[-3 -3 6],[-2 1 1],[-60 -60 120]};
cases(3).inter.zeeman.euler={[0 0 0],[0.3 0.6 0.9],[1.0 0.5 0.2]};
cases(3).inter.coordinates={[0 0 0]; [1.8 0 0]; [0 0 1.09]};
cases(3).inter.coupling.scalar=cell(3); cases(3).inter.coupling.scalar{1,3}=140;
cases(3).inter.relaxation={'redfield'};
cases(3).inter.equilibrium='IME';
cases(3).inter.rlx_keep='secular';
cases(3).inter.temperature=298;
cases(3).inter.tau_c={2e-10};
cases(3).bas=struct('formalism','sphten-liouv','approximation','none');
cases(3).noe=struct('dt',5e-3,'nsteps',200);

% Axial rotational diffusion, dynamic frequency shifts kept, lab-frame retention
cases(4).name='axial_dfs';
cases(4).sys=struct('isotopes',{{'1H','13C'}},'magnet',14.1);
cases(4).inter.zeeman.scalar={1.0 50.0};
cases(4).inter.zeeman.eigs={[],[-40 -40 80]};
cases(4).inter.zeeman.euler={[],[0.2 0.4 0.6]};
cases(4).inter.coordinates={[0 0 0]; [0 0 1.09]};
cases(4).inter.relaxation={'redfield'};
cases(4).inter.equilibrium='zero';
cases(4).inter.rlx_keep='labframe';
cases(4).inter.rlx_dfs='keep';
cases(4).inter.temperature=298;
cases(4).inter.tau_c={[5e-10 1e-10]};
cases(4).bas=struct('formalism','sphten-liouv','approximation','none');
cases(4).noe=struct('dt',2e-3,'nsteps',100);

% Extended T1/T2 model plus anisotropic damping
cases(5).name='t1t2_damp';
cases(5).sys=struct('isotopes',{{'1H','13C'}},'magnet',14.1);
cases(5).inter.zeeman.scalar={1.0 50.0};
cases(5).inter.coupling.scalar=cell(2); cases(5).inter.coupling.scalar{1,2}=140;
cases(5).inter.relaxation={'t1_t2','damp'};
cases(5).inter.equilibrium='dibari';
cases(5).inter.rlx_keep='diagonal';
cases(5).inter.temperature=298;
cases(5).inter.r1_rates={1.0, 2.0};
cases(5).inter.r2_rates={5.0, 6.0};
cases(5).inter.damp_rate=diag([0.3 0.5 0.7]);
cases(5).bas=struct('formalism','sphten-liouv','approximation','none');
cases(5).noe=struct('dt',1e-2,'nsteps',100);

% Settings common to all cases
dump_file=[mfilename('fullpath') '.m'];
angles=[0.3 0.9 1.2];

% Loop over cases
for n=1:numel(cases)

    % Never overwrite existing fixtures: skip cases that are already there
    case_dir=fullfile(out_root,['relax_' cases(n).name]);
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
    sys.disable={'hygiene','asyredf'}; % serial Redfield path; existentials() would reject the isworkernode.m shadow
    inter=cases(n).inter; bas=cases(n).bas; noe=cases(n).noe;
    spin_system=create(sys,inter);
    spin_system=basis(spin_system,bas);
    oracle_write(h5_file,'/bas/basis',spin_system.bas.basis);

    % Relaxation superoperators: isotropic and at an orientation
    progress(case_label,'relaxation');
    R=relaxation(spin_system);
    oracle_write(h5_file,'/rlx/R',R);
    oracle_write(h5_file,'/rlx/R_oriented',relaxation(spin_system,angles));

    % Lab-frame Hamiltonian (left) and thermal equilibrium
    progress(case_label,'equilibrium');
    [HL,QL]=hamiltonian(assume(spin_system,'labframe'),'left');
    oracle_write(h5_file,'/ham/HL',HL);
    rho_eq=equilibrium(spin_system,HL);
    oracle_write(h5_file,'/states/rho_eq',rho_eq);
    oracle_write(h5_file,'/states/rho_eq_oriented',equilibrium(spin_system,HL,QL,angles));

    % Thermalisation of the untreated superoperator (equilibrium 'zero'), both methods
    ss_zero=spin_system; ss_zero.rlx.equilibrium='zero';
    R0=relaxation(ss_zero);
    oracle_write(h5_file,'/rlx/R_zero',R0);
    oracle_write(h5_file,'/rlx/R_ime',thermalize(spin_system,R0,[],[],rho_eq,'IME'));
    oracle_write(h5_file,'/rlx/R_dibari',thermalize(spin_system,R0,HL,inter.temperature,[],'dibari'));

    % Transient NOE: invert spin 1 and detect Lz of every spin
    progress(case_label,'NOE trajectory');
    Lz1=state(spin_system,{'Lz'},{1});
    rho=rho_eq-2*Lz1*(Lz1'*rho_eq)/norm(Lz1)^2;
    nspins=spin_system.comp.nspins; coil=zeros(size(rho,1),nspins);
    for k=1:nspins, coil(:,k)=state(spin_system,{'Lz'},{k}); end
    oracle_write(h5_file,'/states/rho_inverted',rho);
    oracle_write(h5_file,'/noe/multichannel',evolution(spin_system,1i*R,coil,rho,noe.dt,noe.nsteps,'multichannel'));

    % Provenance record
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'noe',noe,'angles',angles);
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
