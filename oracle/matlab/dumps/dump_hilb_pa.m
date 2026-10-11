% Oracle dump for the zeeman-hilb pulse-acquire vertical slice. For each
% case below, runs the chain create -> basis -> operator -> hamiltonian
% -> frqoffset -> state -> propagator -> evolution step by step, and
% then end to end through liquid() and acquire(), writing every inter-
% mediate into <out_root>/hilb_pa_<case>/data.h5 with a meta.json pro-
% venance record. Syntax:
%
%                       dump_hilb_pa(out_root)
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
%       out conventions on the Python side. Progress is printed to the
%       console with a timestamp per stage; the Spinach report output
%       of each case goes to <case directory>/spinach.log.

function dump_hilb_pa(out_root)

% Check consistency
grumble(out_root);

% Strongly coupled homonuclear two-spin system
cases(1).name='ab_2spin';
cases(1).sys.isotopes={'1H','1H'};
cases(1).inter.zeeman.scalar={1.00 1.20};
cases(1).inter.coupling.scalar={0 12.0; 0 0};
cases(1).pars=struct('spins',{{'1H'}},'offset',660,'sweep',500,...
                     'npoints',1024,'zerofill',4096);

% Homonuclear three-spin system with one weakly coupled spin
cases(2).name='abx_3spin';
cases(2).sys.isotopes={'1H','1H','1H'};
cases(2).inter.zeeman.scalar={1.00 1.15 3.00};
cases(2).inter.coupling.scalar={0 15.0 7.0; 0 0 3.0; 0 0 0};
cases(2).pars=struct('spins',{{'1H'}},'offset',1200,'sweep',2000,...
                     'npoints',2048,'zerofill',8192);

% Heteronuclear two-spin system, proton detection
cases(3).name='hc_2spin';
cases(3).sys.isotopes={'1H','13C'};
cases(3).inter.zeeman.scalar={2.00 30.0};
cases(3).inter.coupling.scalar={0 145.0; 0 0};
cases(3).pars=struct('spins',{{'1H'}},'offset',1200,'sweep',500,...
                     'npoints',1024,'zerofill',4096);

% Spin-1/2 coupled to spin-1, proton detection
cases(4).name='hd_2spin';
cases(4).sys.isotopes={'1H','2H'};
cases(4).inter.zeeman.scalar={4.50 4.50};
cases(4).inter.coupling.scalar={0 43.0; 0 0};
cases(4).pars=struct('spins',{{'1H'}},'offset',2700,'sweep',400,...
                     'npoints',1024,'zerofill',4096);

% Settings common to all cases
dump_file=[mfilename('fullpath') '.m'];
bas.formalism='zeeman-hilb';
bas.approximation='none';

% Loop over cases
for n=1:numel(cases)

    % Never overwrite existing fixtures: skip cases that are already there
    case_dir=fullfile(out_root,['hilb_pa_' cases(n).name]);
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
    sys=cases(n).sys; sys.magnet=14.1; sys.output=fullfile(case_dir,'spinach.log');
    sys.disable={'hygiene'}; % existentials() would reject the isworkernode.m shadow
    inter=cases(n).inter; pars=cases(n).pars;
    spin_system=create(sys,inter);
    spin_system=basis(spin_system,bas);
    nspins=spin_system.comp.nspins;
    oracle_write(h5_file,'/comp/mults',spin_system.comp.mults);
    oracle_write(h5_file,'/inter/basefrqs',spin_system.inter.basefrqs);
    oracle_write(h5_file,'/inter/zeeman_matrix',spin_system.inter.zeeman.matrix);
    oracle_write(h5_file,'/inter/coupling_matrix',spin_system.inter.coupling.matrix);

    % Single-spin operators
    progress(case_label,'operators');
    op_names={'E','Lz','Lx','Ly','L+','L-'};
    op_keys={'E','Lz','Lx','Ly','Lp','Lm'};
    for k=1:numel(op_names)
        single_ops=cell(1,nspins);
        for m=1:nspins
            single_ops{m}=operator(spin_system,op_names(k),{m});
        end
        oracle_write(h5_file,['/ops/' op_keys{k}],single_ops);
    end

    % Product and isotope-sum operators
    oracle_write(h5_file,'/ops/LzLp_12',operator(spin_system,{'Lz','L+'},{1,2}));
    oracle_write(h5_file,'/ops/Lz_det',operator(spin_system,'Lz',pars.spins{1}));
    oracle_write(h5_file,'/ops/Lp_det',operator(spin_system,'L+',pars.spins{1}));

    % Hamiltonian and its Zeeman and coupling parts
    progress(case_label,'Hamiltonian');
    H=hamiltonian(assume(spin_system,'nmr'));
    oracle_write(h5_file,'/ham/H',H);
    oracle_write(h5_file,'/ham/H_zeeman',hamiltonian(assume(spin_system,'nmr','zeeman')));
    oracle_write(h5_file,'/ham/H_coupling',hamiltonian(assume(spin_system,'nmr','couplings')));

    % Transmitter offset
    H=frqoffset(spin_system,H,pars);
    oracle_write(h5_file,'/ham/H_off',H);

    % Initial and detection states
    rho0=state(spin_system,'L+',pars.spins{1});
    coil=state(spin_system,'L+',pars.spins{1});
    oracle_write(h5_file,'/states/rho0',rho0);
    oracle_write(h5_file,'/states/coil',coil);

    % Propagator over one dwell time
    progress(case_label,'propagator');
    P=propagator(spin_system,H,1/pars.sweep);
    oracle_write(h5_file,'/prop/P',P);

    % Explicit time stepping with the propagator
    progress(case_label,'FID by explicit stepping');
    fid=zeros([pars.npoints 1],'like',1i); rho=rho0;
    for k=1:pars.npoints
        fid(k)=hdot(coil,rho); rho=P*rho*P';
    end
    oracle_write(h5_file,'/fid/explicit',fid);

    % Kernel evolution function
    progress(case_label,'FID by evolution()');
    fid=evolution(spin_system,H,coil,rho0,1/pars.sweep,pars.npoints-1,'observable');
    oracle_write(h5_file,'/fid/evolution',fid);

    % End-to-end context and sequence
    progress(case_label,'FID by liquid() and acquire()');
    seq_pars=pars; seq_pars.rho0=rho0; seq_pars.coil=coil; seq_pars.decouple={};
    fid=liquid(spin_system,@acquire,seq_pars,'nmr');
    oracle_write(h5_file,'/fid/liquid',fid);

    % Unapodised spectrum
    oracle_write(h5_file,'/spectrum/raw',fftshift(fft(fid,pars.zerofill)));

    % Provenance record
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'pars',pars,...
                  'rho0','L+ on detected isotope','coil','L+ on detected isotope',...
                  'assumptions','nmr','timestep',1/pars.sweep);
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

