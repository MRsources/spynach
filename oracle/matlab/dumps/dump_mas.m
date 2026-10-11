% Oracle dump for magic angle spinning in Fokker-Planck form: the rotor
% phase grid and spectral derivative, the Fokker-Planck generator at the
% first powder orientation, the expanded states, and the singlerot()
% pulse-acquire FID. Syntax:
%
%                       dump_mas(out_root)
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

function dump_mas(out_root)

% Check consistency
grumble(out_root);

% CSA pair under MAS (examples/nmr_solids/mas_powder_csa_fplanck.m, smaller grid and rank)
cases(1).name='csa';
cases(1).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(1).inter.zeeman.eigs={[-2 -2 4]-5,[-1 -3 4]+5};
cases(1).inter.zeeman.euler={[0 0 0],[0.3 0.9 1.2]};
cases(1).bas=struct('formalism','sphten-liouv','approximation','none','projections',{{1}});
cases(1).pars=struct('spins',{{'1H'}},'offset',0,'rate',500,'axis',[1 1 1],'max_rank',7,...
                     'grid','leb_2ang_rank_11','sweep',2e4,'npoints',128,'zerofill',512);

% Dipolar pair under MAS (examples/nmr_solids/mas_powder_dip_fplanck.m)
cases(2).name='dip';
cases(2).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(2).inter.zeeman.scalar={5.0,-2.0};
cases(2).inter.coordinates={[0 0 0]; [0 3.9 0.1]};
cases(2).bas=struct('formalism','sphten-liouv','approximation','none','projections',{{1}});
cases(2).pars=struct('spins',{{'1H'}},'offset',0,'rate',1000,'axis',[1 1 1],'max_rank',7,...
                     'grid','leb_2ang_rank_11','sweep',2e4,'npoints',128,'zerofill',512);

% Deuterium quadrupole under MAS (examples/nmr_solids/mas_powder_nqi_fplanck.m)
cases(3).name='nqi';
cases(3).sys=struct('isotopes',{{'2H'}},'magnet',9.4);
cases(3).inter.coupling.eigs={[-1e3 -2e3 3e3]};
cases(3).inter.coupling.euler={[0 0 0]};
cases(3).bas=struct('formalism','sphten-liouv','approximation','none','projections',{{1}});
cases(3).pars=struct('spins',{{'2H'}},'offset',0,'rate',1000,'axis',[1 1 1],'max_rank',7,...
                     'grid','leb_2ang_rank_11','sweep',2e4,'npoints',128,'zerofill',512);

% Single crystal, off-magic-angle axis, with an offset
cases(4).name='crystal_offaxis';
cases(4).sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
cases(4).inter.zeeman.eigs={[-2 -2 4]-5,[-1 -3 4]+5};
cases(4).inter.zeeman.euler={[0 0 0],[0.3 0.9 1.2]};
cases(4).bas=struct('formalism','sphten-liouv','approximation','none','projections',{{1}});
cases(4).pars=struct('spins',{{'1H'}},'offset',300,'rate',800,'axis',[0.5 0.2 1],'max_rank',5,...
                     'grid','single_crystal','sweep',2e4,'npoints',128,'zerofill',512);

% Settings common to all cases
dump_file=[mfilename('fullpath') '.m'];

% Loop over cases
for n=1:numel(cases)

    % Never overwrite existing fixtures: skip cases that are already there
    case_dir=fullfile(out_root,['mas_' cases(n).name]);
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
    oracle_write(h5_file,'/bas/basis',spin_system.bas.basis);

    % Rotor phase grid and spectral derivative
    spc_dim=2*pars.max_rank+1;
    [rotor_phases,d_dphi]=fourdif(spc_dim,1);
    oracle_write(h5_file,'/fp/rotor_phases',rotor_phases);
    oracle_write(h5_file,'/fp/d_dphi',d_dphi);

    % Fokker-Planck generator at the first orientation of the grid, as singlerot() builds it
    progress(case_label,'Fokker-Planck generator');
    g=load([spin_system.sys.root_dir filesep 'kernel' filesep 'grids' filesep pars.grid],'alphas','betas','gammas','weights');
    [rotor_phi,rotor_theta,~]=cart2sph(pars.axis(1),pars.axis(2),pars.axis(3)); rotor_theta=pi/2-rotor_theta;
    nmr=assume(spin_system,'nmr');
    [I,Q]=hamiltonian(nmr); I=frqoffset(spin_system,I,pars);
    spn_dim=size(I,2);
    H=cell(spc_dim,1); H(:)={I};
    for k=1:spc_dim
        for r=1:numel(Q)
            D_comp=wigner(r,rotor_phi,rotor_theta,0)*wigner(r,0,0,rotor_phases(k))*wigner(r,g.alphas(1),g.betas(1),g.gammas(1));
            for p=1:(2*r+1)
                for q=1:(2*r+1)
                    H{k}=H{k}+D_comp(p,q)*Q{r}{p,q};
                end
            end
        end
        H{k}=sparse(H{k});
    end
    M=2*pi*pars.rate*kron(d_dphi,speye([spn_dim spn_dim]));
    G=clean_up(spin_system,blkdiag(H{:})+1i*M,spin_system.tols.liouv_zero);
    oracle_write(h5_file,'/fp/G_first_orientation',G);
    oracle_write(h5_file,'/fp/first_orientation',[g.alphas(1) g.betas(1) g.gammas(1)]);
    oracle_write(h5_file,'/fp/rotor_angles',[rotor_phi rotor_theta]);

    % Expanded states, by running a probe sequence through the context at the single crystal grid
    rho0=state(spin_system,'L+',det); coil=state(spin_system,'L+',det);
    seq_pars=pars; seq_pars.rho0=rho0; seq_pars.coil=coil; seq_pars.decouple={}; seq_pars.verbose=0; seq_pars.serial=1;
    probe_pars=seq_pars; probe_pars.grid='single_crystal';
    states=singlerot(spin_system,@probe_states,probe_pars,'nmr');
    oracle_write(h5_file,'/states/rho0_fp_crystal',states{1});
    oracle_write(h5_file,'/states/coil_fp',states{2});

    % Pulse-acquire FID through singlerot()
    progress(case_label,'singlerot FID');
    fid=singlerot(spin_system,@acquire,seq_pars,'nmr');
    oracle_write(h5_file,'/fid/singlerot',fid);
    oracle_write(h5_file,'/spectrum/raw',fftshift(fft(fid,pars.zerofill)));

    % Provenance record
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'pars',pars,'assumptions','nmr',...
                  'rho0','L+ on detected isotope','coil','L+ on detected isotope');
    oracle_meta(case_dir,dump_file,spin_system,inputs);

    % Close the Spinach log
    fclose(spin_system.sys.output);
    progress(case_label,'finished');

end

end

% Probe sequence returning the expanded initial and detection states
function answer=probe_states(~,parameters,~,~,~)
answer={parameters.rho0,parameters.coil};
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
