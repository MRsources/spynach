% Oracle dump for anisotropic interactions: interaction tensors built from
% eigenvalues and Euler angles, from coordinates (dipolar) and from quad-
% rupolar parameters, the isotropic Hamiltonian and its irreducible com-
% ponents Q, the Hamiltonian at two orientations, the spherical grids,
% and single-crystal and powder-averaged pulse-acquire FIDs. Syntax:
%
%                       dump_aniso(out_root)
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

function dump_aniso(out_root)

% Check consistency
grumble(out_root);

% Two protons with CSA tensors (examples/nmr_solids/static_powder_csa.m)
cases(1).name='csa_pair';
cases(1).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(1).inter.zeeman.eigs={[-2 -2 4]-5,[-1 -3 4]+5};
cases(1).inter.zeeman.euler={[0 0 0],[0.3 0.9 1.2]};
cases(1).bas=struct('formalism','sphten-liouv','approximation','none','projections',{{1}});
cases(1).pars=struct('spins',{{'1H'}},'offset',0,'sweep',15000,'npoints',128,'zerofill',512);
cases(1).assumptions='nmr';

% Dipolar-coupled proton pair from coordinates (examples/nmr_solids/static_powder_dip.m)
cases(2).name='dip_pair';
cases(2).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(2).inter.zeeman.scalar={5.0,-2.0};
cases(2).inter.coordinates={[0 0 0]; [0 3.9 0.1]};
cases(2).bas=struct('formalism','sphten-liouv','approximation','none','projections',{{1}});
cases(2).pars=struct('spins',{{'1H'}},'offset',1000,'sweep',12000,'npoints',128,'zerofill',512);
cases(2).assumptions='nmr';

% Two 14N with quadrupolar couplings, Hilbert space (examples/nmr_solids/static_powder_nqi_a.m)
cases(3).name='nqi_14n';
cases(3).sys=struct('isotopes',{{'14N','14N'}},'magnet',21.1);
cases(3).inter.coupling.matrix=cell(2);
cases(3).inter.coupling.matrix{1,1}=eeqq2nqi(1.24e6,0.22,1,[0 0 0]);
cases(3).inter.coupling.matrix{2,2}=eeqq2nqi(3.06e6,0.40,1,[0.2 0.4 0.6]);
cases(3).bas=struct('formalism','zeeman-hilb','approximation','none');
cases(3).pars=struct('spins',{{'14N'}},'offset',0,'sweep',6e6,'npoints',128,'zerofill',512);
cases(3).assumptions='nmr';

% 1H-13C with tensor shielding and an anisotropic J coupling, Hilbert space
cases(4).name='hc_tensors';
cases(4).sys=struct('isotopes',{{'1H','13C'}},'magnet',14.1);
cases(4).inter.zeeman.eigs={[-2 -2 4]+3,[20 60 100]};
cases(4).inter.zeeman.euler={[0.1 0.2 0.3],[0.4 0.5 0.6]};
cases(4).inter.coupling.eigs=cell(2); cases(4).inter.coupling.eigs{1,2}=[100 120 200];
cases(4).inter.coupling.euler=cell(2); cases(4).inter.coupling.euler{1,2}=[0.7 0.8 0.9];
cases(4).inter.coordinates={[0 0 0]; [0 0 1.09]};
cases(4).bas=struct('formalism','zeeman-hilb','approximation','none');
cases(4).pars=struct('spins',{{'13C'}},'offset',9000,'sweep',1e5,'npoints',128,'zerofill',512);
cases(4).assumptions='nmr';

% Nitroxide-like electron with a 14N hyperfine tensor, ESR assumptions, Liouville space
cases(5).name='esr_nitroxide';
cases(5).sys=struct('isotopes',{{'E','14N'}},'magnet',0.33);
cases(5).inter.zeeman.eigs={[2.0089 2.0061 2.0027],[]};
cases(5).inter.zeeman.euler={[0 0 0],[]};
cases(5).inter.coupling.eigs=cell(2); cases(5).inter.coupling.eigs{1,2}=[16e6 16e6 86e6];
cases(5).inter.coupling.euler=cell(2); cases(5).inter.coupling.euler{1,2}=[0 0 0];
cases(5).bas=struct('formalism','sphten-liouv','approximation','none');
cases(5).pars=struct('spins',{{'E'}},'offset',0,'sweep',1e9,'npoints',128,'zerofill',512);
cases(5).assumptions='esr';

% Settings common to all cases
dump_file=[mfilename('fullpath') '.m'];
orientations={[0 0 0],[0.3 0.9 1.2]};
grids={'single_crystal','leb_2ang_rank_11','rep_2ang_100pts_sph'};

% Loop over cases
for n=1:numel(cases)

    % Never overwrite existing fixtures: skip cases that are already there
    case_dir=fullfile(out_root,['aniso_' cases(n).name]);
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
    oracle_write(h5_file,'/inter/zeeman_matrix',spin_system.inter.zeeman.matrix);
    oracle_write(h5_file,'/inter/coupling_matrix',spin_system.inter.coupling.matrix);
    oracle_write(h5_file,'/inter/proxmatrix',spin_system.inter.proxmatrix);
    oracle_write(h5_file,'/bas/basis',spin_system.bas.basis);

    % Isotropic Hamiltonian and irreducible components
    progress(case_label,'Hamiltonian and Q');
    assumed=assume(spin_system,cases(n).assumptions);
    [I,Q]=hamiltonian(assumed);
    oracle_write(h5_file,'/ham/I',I);
    oracle_write(h5_file,'/ham/Q1',Q{1});
    oracle_write(h5_file,'/ham/Q2',Q{2});
    I=frqoffset(spin_system,I,pars);
    oracle_write(h5_file,'/ham/I_off',I);
    for k=1:numel(orientations)
        H=I+orientation(Q,orientations{k}); H=(H+H')/2;
        oracle_write(h5_file,['/ham/H_orient_' int2str(k)],H);
    end
    for k=1:3
        oracle_write(h5_file,['/conv/wigner_' int2str(k)],wigner(k,0.3,0.9,1.2));
    end

    % Grids used below, so that the converted copies can be checked
    for k=1:numel(grids)
        g=load([spin_system.sys.root_dir filesep 'kernel' filesep 'grids' filesep grids{k}],...
               'alphas','betas','gammas','weights');
        oracle_write(h5_file,['/grids/' grids{k}],[g.alphas(:) g.betas(:) g.gammas(:) g.weights(:)]);
    end

    % Initial and detection states
    rho0=state(spin_system,'L+',det); coil=state(spin_system,'L+',det);
    oracle_write(h5_file,'/states/rho0',rho0);

    % Single crystal and powder FIDs through the contexts
    progress(case_label,'crystal and powder');
    seq_pars=pars; seq_pars.rho0=rho0; seq_pars.coil=coil; seq_pars.decouple={}; seq_pars.verbose=0;
    seq_pars.orientation=orientations{2};
    fid=crystal(spin_system,@acquire,seq_pars,cases(n).assumptions);
    oracle_write(h5_file,'/fid/crystal',fid);
    seq_pars=rmfield(seq_pars,'orientation');
    seq_pars.grid='leb_2ang_rank_11'; seq_pars.serial=1;
    fid=powder(spin_system,@acquire,seq_pars,cases(n).assumptions);
    oracle_write(h5_file,'/fid/powder_leb11',fid);
    seq_pars.grid='rep_2ang_100pts_sph';
    fid=powder(spin_system,@acquire,seq_pars,cases(n).assumptions);
    oracle_write(h5_file,'/fid/powder_rep100',fid);
    oracle_write(h5_file,'/spectrum/powder_rep100',fftshift(fft(fid,pars.zerofill)));

    % Provenance record
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'pars',pars,'assumptions',cases(n).assumptions,...
                  'orientations',{orientations},'grids',{grids},'rho0','L+ on detected isotope','coil','L+ on detected isotope');
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
