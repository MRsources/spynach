% Oracle dump for the pulse sequence library: processing (apodisation),
% analytical decoupling and homospoil, the refocus evolution output,
% acquire() with dead time and decoupling, and the cosy, noesy and hsqc
% sequences through liquid(), including the isotopomer sum of dilute().
% Syntax:
%
%                       dump_seq(out_root)
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

function dump_seq(out_root)

% Check consistency
grumble(out_root);
dump_file=[mfilename('fullpath') '.m'];
bas=struct('formalism','sphten-liouv','approximation','none');

%% Case 1: COSY-90 on the Derome three-spin system, with processing helpers
case_dir=fullfile(out_root,'seq_cosy_derome');
if isfolder(case_dir)
    fprintf('%s [cosy_derome] exists, skipped\n',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')));
else
    mkdir(case_dir); h5_file=fullfile(case_dir,'data.h5'); progress('cosy_derome 1/3','started');
    probes(h5_file);
    sys=struct('isotopes',{{'1H','1H','1H'}},'magnet',16.1);
    sys.output=fullfile(case_dir,'spinach.log'); sys.disable={'hygiene'};
    inter.zeeman.scalar={3.70 3.92 4.50};
    inter.coupling.scalar=cell(3); inter.coupling.scalar{1,2}=10;
    inter.coupling.scalar{2,3}=12; inter.coupling.scalar{1,3}=4;
    spin_system=create(sys,inter); spin_system=basis(spin_system,bas);
    pars=struct('angle',pi/2,'offset',2800,'sweep',700,'npoints',[64 64],'zerofill',[128 128],'spins',{{'1H'}});
    oracle_write(h5_file,'/bas/basis',spin_system.bas.basis);
    progress('cosy_derome 1/3','cosy');
    fid=liquid(spin_system,@cosy,pars,'nmr');
    oracle_write(h5_file,'/fid/cosy',fid);
    fid_apod=apodisation(spin_system,fid,{{'sqcos'},{'sqcos'}});
    oracle_write(h5_file,'/fid/cosy_apodised',fid_apod);
    oracle_write(h5_file,'/spectrum/cosy',fftshift(fft2(fid_apod,pars.zerofill(2),pars.zerofill(1))));
    % Processing helpers on a known vector and matrix
    v=(1:16)'+1i*(16:-1:1)'; A=reshape(1:48,[6 8])+1i;
    oracle_write(h5_file,'/apod/exp6',apodisation(spin_system,v,{{'exp',6}}));
    oracle_write(h5_file,'/apod/gauss4_nohalf',apodisation(spin_system,v,{{'gauss',4}},false));
    oracle_write(h5_file,'/apod/crisp',apodisation(spin_system,v,{{'crisp'}}));
    oracle_write(h5_file,'/apod/kaiser3',apodisation(spin_system,v,{{'kaiser',3}}));
    oracle_write(h5_file,'/apod/matrix_sqcos_sin',apodisation(spin_system,A,{{'sqcos'},{'sin'}}));
    oracle_write(h5_file,'/apod/matrix_none_cos',apodisation(spin_system,A,{{'none'},{'cos'}}));
    % Refocus output and acquire with dead time, on the same Liouvillian
    nmr=assume(spin_system,'nmr'); H=frqoffset(spin_system,hamiltonian(nmr),pars);
    rho=state(spin_system,'L+','1H');
    stack=evolution(spin_system,H,[],rho,1e-3,7,'trajectory');
    oracle_write(h5_file,'/evo/trajectory',stack);
    oracle_write(h5_file,'/evo/refocus',evolution(spin_system,H,[],stack,1e-3,[],'refocus'));
    apars=struct('spins',{{'1H'}},'offset',2800,'sweep',700,'npoints',32,'dead_time',3e-3,'decouple',{{}});
    apars.rho0=rho; apars.coil=rho;
    oracle_write(h5_file,'/fid/acquire_dead_time',liquid(spin_system,@acquire,apars,'nmr'));
    oracle_write(h5_file,'/states/homospoil_destroy',homospoil(spin_system,stack,'destroy'));
    oracle_write(h5_file,'/states/homospoil_keep',homospoil(spin_system,stack,'keep'));
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'pars',pars,'apars',rmfield(rmfield(apars,'rho0'),'coil'));
    oracle_meta(case_dir,dump_file,spin_system,inputs); fclose(spin_system.sys.output);
    progress('cosy_derome 1/3','finished');
end

%% Case 2: NOESY on a dipolar proton pair with Redfield relaxation
case_dir=fullfile(out_root,'seq_noesy_pair');
if isfolder(case_dir)
    fprintf('%s [noesy_pair] exists, skipped\n',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')));
else
    mkdir(case_dir); h5_file=fullfile(case_dir,'data.h5'); progress('noesy_pair 2/3','started');
    probes(h5_file);
    sys=struct('isotopes',{{'1H','1H'}},'magnet',14.1);
    sys.output=fullfile(case_dir,'spinach.log'); sys.disable={'hygiene','asyredf'};
    inter=struct();
    inter.zeeman.scalar={1.0 3.0};
    inter.coordinates={[0 0 0]; [0 0 2.5]};
    inter.relaxation={'redfield'}; inter.equilibrium='IME'; inter.rlx_keep='kite';
    inter.temperature=298; inter.tau_c={2e-9};
    spin_system=create(sys,inter); spin_system=basis(spin_system,bas);
    pars=struct('tmix',0.3,'offset',1200,'sweep',[2000 2000],'npoints',[48 48],'zerofill',[128 128],...
                'spins',{{'1H'}},'needs',{{'rho_eq'}});
    oracle_write(h5_file,'/bas/basis',spin_system.bas.basis);
    progress('noesy_pair 2/3','noesy');
    fid=liquid(spin_system,@noesy,pars,'nmr');
    oracle_write(h5_file,'/fid/noesy_cos',fid.cos);
    oracle_write(h5_file,'/fid/noesy_sin',fid.sin);
    opars=pars; opars.oldschool=true();
    fid_old=liquid(spin_system,@noesy,opars,'nmr');
    oracle_write(h5_file,'/fid/noesy_oldschool_cos',fid_old.cos);
    oracle_write(h5_file,'/states/rho_eq',equilibrium(spin_system));
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'pars',pars);
    oracle_meta(case_dir,dump_file,spin_system,inputs); fclose(spin_system.sys.output);
    progress('noesy_pair 2/3','finished');
end

%% Case 3: HSQC with isotopomer dilution on a 1H,13C,13C,1H system
case_dir=fullfile(out_root,'seq_hsqc_dilute');
if isfolder(case_dir)
    fprintf('%s [hsqc_dilute] exists, skipped\n',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')));
else
    mkdir(case_dir); h5_file=fullfile(case_dir,'data.h5'); progress('hsqc_dilute 3/3','started');
    probes(h5_file);
    sys=struct('isotopes',{{'1H','13C','13C','1H'}},'magnet',5.9);
    sys.output=fullfile(case_dir,'spinach.log'); sys.disable={'hygiene'};
    inter=struct();
    inter.zeeman.scalar={3.0 40.0 50.0 1.0};
    inter.coupling.scalar=cell(4); inter.coupling.scalar{1,2}=140; inter.coupling.scalar{3,4}=150;
    inter.coupling.scalar{2,3}=35; inter.coupling.scalar{1,4}=7;
    spin_system=create(sys,inter);
    pars=struct('J',140,'sweep',[3000 2000],'offset',[1000 500],'npoints',[32 32],'zerofill',[128 128],...
                'spins',{{'13C','1H'}},'decouple_f1',{{'1H'}},'decouple_f2',{{'13C'}});
    subsystems=dilute(spin_system,'13C');
    spectrum=zeros(pars.zerofill(2),pars.zerofill(1),'like',1i);
    for n=1:numel(subsystems)
        subsystem=basis(subsystems{n},bas);
        oracle_write(h5_file,['/sub/isotopes_' int2str(n)],cellfun(@(x)double(x(1)),subsystem.comp.isotopes));
        oracle_write(h5_file,['/sub/basis_' int2str(n)],subsystem.bas.basis);
        progress('hsqc_dilute 3/3',['hsqc subsystem ' int2str(n)]);
        fid=liquid(subsystem,@hsqc,pars,'nmr');
        oracle_write(h5_file,['/fid/hsqc_pos_' int2str(n)],fid.pos);
        oracle_write(h5_file,['/fid/hsqc_neg_' int2str(n)],fid.neg);
        fid.pos=apodisation(spin_system,fid.pos,{{'sqcos'},{'sqcos'}});
        fid.neg=apodisation(spin_system,fid.neg,{{'sqcos'},{'sqcos'}});
        f1_pos=fftshift(fft(fid.pos,pars.zerofill(2),1),1);
        f1_neg=fftshift(fft(fid.neg,pars.zerofill(2),1),1);
        spectrum=spectrum+fftshift(fft(f1_pos+conj(f1_neg),pars.zerofill(1),2),2);
    end
    oracle_write(h5_file,'/spectrum/hsqc',spectrum);
    % Decoupling of a Liouvillian and a state on the full system
    full_system=basis(spin_system,bas); nmr=assume(full_system,'nmr'); H=hamiltonian(nmr);
    rho=state(full_system,'L+','1H')+state(full_system,{'L+','Lz'},{1,2});
    [Hd,rhod]=decouple(full_system,H,rho,{'13C'});
    oracle_write(h5_file,'/decouple/H',H); oracle_write(h5_file,'/decouple/H_dec',Hd);
    oracle_write(h5_file,'/decouple/rho',rho); oracle_write(h5_file,'/decouple/rho_dec',rhod);
    inputs=struct('sys',sys,'inter',inter,'bas',bas,'pars',pars,'nsub',numel(subsystems));
    oracle_meta(case_dir,dump_file,full_system,inputs); fclose(full_system.sys.output);
    progress('hsqc_dilute 3/3','finished');
end

end

% Layout convention probes with known content
function probes(h5_file)
oracle_write(h5_file,'/probe/dense',reshape(1:24,[2 3 4])+1i*reshape(101:124,[2 3 4]));
oracle_write(h5_file,'/probe/column',[1; 2; 3]);
oracle_write(h5_file,'/probe/sparse',sparse([1 3 2],[2 1 4],[5+1i 7 -2i],3,4));
oracle_write(h5_file,'/probe/cell',{[1 2],sparse(eye(2)); [],7});
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
