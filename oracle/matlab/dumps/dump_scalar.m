% Oracle dump for scalar relaxation of the first kind (SRFK, scalar
% couplings modulated by chemical exchange) and of the second kind (SRSK,
% fast quadrupolar relaxation of a source spin transferred through J).
% Syntax:
%
%                       dump_scalar(out_root)
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

function dump_scalar(out_root)

% Check consistency
grumble(out_root);

% SRFK alone on a two-proton system (examples/relaxation_theory/scalar_relaxation_1.m)
cases(1).name='srfk_pair';
cases(1).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(1).inter.zeeman.scalar={0.0 2.0};
cases(1).inter.relaxation={'SRFK'};
cases(1).inter.rlx_keep='kite';
cases(1).inter.equilibrium='zero';
cases(1).inter.srfk_tau_c={[1.0 1e-3]};
cases(1).inter.srfk_mdepth=cell(2);
cases(1).inter.srfk_mdepth{1,2}=15.0;
cases(1).bas=struct('formalism','sphten-liouv','approximation','none');
cases(1).noe=struct('dt',1e-2,'nsteps',100);

% SRSK from a fast-relaxing quadrupolar 14N onto a scalar-coupled proton
cases(2).name='srsk_hn';
cases(2).sys=struct('isotopes',{{'1H','14N'}},'magnet',11.75);
cases(2).inter.zeeman.scalar={0.0 0.0};
cases(2).inter.coordinates={[0 0 0]; [0 0 1.0]};
cases(2).inter.coupling.scalar=cell(2); cases(2).inter.coupling.scalar{1,2}=50;
cases(2).inter.coupling.eigs=cell(2); cases(2).inter.coupling.eigs{2,2}=[1e6 1e6 -2e6];
cases(2).inter.coupling.euler=cell(2); cases(2).inter.coupling.euler{2,2}=[0 0 0];
cases(2).inter.relaxation={'redfield','SRSK'};
cases(2).inter.srsk_sources=2;
cases(2).inter.rlx_keep='secular';
cases(2).inter.equilibrium='zero';
cases(2).inter.tau_c={1e-9};
cases(2).bas=struct('formalism','sphten-liouv','approximation','none');
cases(2).noe=struct('dt',1e-3,'nsteps',100);

% Both kinds together with Redfield theory, CSA and a quadrupolar tensor at an orientation
cases(3).name='srfk_srsk_3spin';
cases(3).sys=struct('isotopes',{{'1H','1H','14N'}},'magnet',11.75);
cases(3).inter.zeeman.scalar={1.0 2.0 0.0};
cases(3).inter.zeeman.eigs={[-3 -3 6],[],[-100 -100 200]};
cases(3).inter.zeeman.euler={[0 0 0],[],[0.3 0.6 0.9]};
cases(3).inter.coordinates={[0 0 0]; [1.8 0 0]; [0 0 1.0]};
cases(3).inter.coupling.scalar=cell(3);
cases(3).inter.coupling.scalar{1,2}=10; cases(3).inter.coupling.scalar{1,3}=40; cases(3).inter.coupling.scalar{2,3}=5;
cases(3).inter.coupling.eigs=cell(3); cases(3).inter.coupling.eigs{3,3}=[1e6 1e6 -2e6];
cases(3).inter.coupling.euler=cell(3); cases(3).inter.coupling.euler{3,3}=[0.3 0.6 0.9];
cases(3).inter.relaxation={'redfield','SRFK','SRSK'};
cases(3).inter.srsk_sources=3;
cases(3).inter.rlx_keep='kite';
cases(3).inter.equilibrium='zero';
cases(3).inter.tau_c={1e-9};
cases(3).inter.srfk_tau_c={[1.0 1e-3]};
cases(3).inter.srfk_mdepth=cell(3);
cases(3).inter.srfk_mdepth{1,3}=15.0;
cases(3).bas=struct('formalism','sphten-liouv','approximation','none');
cases(3).noe=struct('dt',1e-2,'nsteps',100);

% Settings common to all cases
dump_file=[mfilename('fullpath') '.m'];

% Loop over cases
for n=1:numel(cases)

    % Never overwrite existing fixtures: skip cases that are already there
    case_dir=fullfile(out_root,['scalar_' cases(n).name]);
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

    % Full relaxation superoperator
    progress(case_label,'relaxation');
    R=relaxation(spin_system);
    oracle_write(h5_file,'/rlx/R',R);

    % The same system without the scalar relaxation theories, when any theory remains
    base_theories=setdiff(inter.relaxation,{'SRFK','SRSK'},'stable');
    if ~isempty(base_theories)
        inter_base=inter; inter_base.relaxation=base_theories;
        for f={'srfk_tau_c','srfk_mdepth','srsk_sources'}
            if isfield(inter_base,f{1}), inter_base=rmfield(inter_base,f{1}); end
        end
        ss_base=basis(create(sys,inter_base),bas);
        oracle_write(h5_file,'/rlx/R_base',relaxation(ss_base));
        fclose(ss_base.sys.output);
    end

    % Lab-frame Hamiltonian (left) and thermal equilibrium
    progress(case_label,'equilibrium');
    HL=hamiltonian(assume(spin_system,'labframe'),'left');
    oracle_write(h5_file,'/ham/HL',HL);
    rho_eq=equilibrium(spin_system,HL);
    oracle_write(h5_file,'/states/rho_eq',rho_eq);

    % Transient NOE: invert spin 1 and detect Lz of every spin
    progress(case_label,'NOE trajectory');
    Lz1=state(spin_system,{'Lz'},{1});
    rho=rho_eq-2*Lz1*(Lz1'*rho_eq)/norm(Lz1)^2;
    nspins=spin_system.comp.nspins; coil=zeros(size(rho,1),nspins);
    for k=1:nspins, coil(:,k)=state(spin_system,{'Lz'},{k}); end
    oracle_write(h5_file,'/states/rho_inverted',rho);
    oracle_write(h5_file,'/noe/multichannel',evolution(spin_system,1i*R,coil,rho,noe.dt,noe.nsteps,'multichannel'));

    % Provenance record
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'noe',noe);
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
