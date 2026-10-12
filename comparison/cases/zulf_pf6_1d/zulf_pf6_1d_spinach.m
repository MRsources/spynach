% Comparison case: ZULF spectrum of PF6-, user_ml/ZULF_PF6_1D.m (Florin
% Teleanu, 2024) with timed stages and the adaptations for Spinach 2.13
% listed in examples/user_ml/zulf_pf6_1d.py (coordinates cell scaling,
% apodisation instead of the legacy apodization). Stage labels must match
% zulf_pf6_1d_spinpy.py. Syntax:
%
%     [data,figs]=zulf_pf6_1d_spinach(timer,options)
%
% Parameters:
%
%     timer   - cmp_timer object
%
%     options - struct, may be empty; npoints and zerofill (default
%               4*4096), rho_file (default user_ml/rho_adiabatic_drop.csv
%               in the repository)
%
% Outputs:
%
%     data    - rho0 (the state read from the file), rho_pulse, fid,
%               spectrum, coil_x, coil_z
%
%     figs    - spectrum: plot_1d of abs(spectrum)

function [data,figs]=zulf_pf6_1d_spinach(timer,options)

if nargin<2, options=struct(); end
npoints=cmp_option(options,'npoints',4*4096);
zerofill=cmp_option(options,'zerofill',4*4096);
repo_dir=fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
rho_file=cmp_option(options,'rho_file',fullfile(repo_dir,'user_ml','rho_adiabatic_drop.csv'));

% Solenoid Field
sys.magnet=2.7*1e-6;
sys.enable={'greedy'};
if cmp_option(options,'gpu',false), sys.enable=[sys.enable {'gpu'}]; end

% Spin system
sys.isotopes={'31P','19F','19F','19F','19F','19F','19F'};

% 1J_PF coupling system for 2M LiPF6
J_PF=710; rPF=1.6;
inter.coupling.scalar=cell(7,7);
for k=2:7, inter.coupling.scalar{1,k}=J_PF; end
inter.coordinates=cellfun(@(c) rPF*c,{[0 0 0],[-1 0 0],[1 0 0],[0 -1 0],[0 1 0],[0 0 -1],[0 0 1]},'UniformOutput',false);

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='none';

% Relaxation theory (validated by create; R is never built, as in the script)
inter.relaxation={'redfield'};
inter.equilibrium='dibari';
inter.temperature=298;
inter.rlx_keep='labframe';
inter.tau_c={0.065e-9};

% Spinach housekeeping
timer.start('create');
spin_system=create(sys,inter);
timer.stop();
timer.start('basis');
spin_system=basis(spin_system,bas);
timer.stop();
timer.start('hamiltonian');
H=hamiltonian(assume(spin_system,'labframe'),'comm');
timer.stop();

% Density operator after the adiabatic drop
rho=csvread(rho_file);
data.rho0=rho;

% Sequence parameters
parameters.sweep=6000;
parameters.npoints=npoints;
parameters.zerofill=zerofill;
parameters.offset=0;
parameters.spins={'19F'};
parameters.axis_units='Hz';
parameters.invert_axis=0;
parameters.detection='quadrature';

timer.start('operators');

% Magnetogyric ratio weights relative to 19F
weights=spin_system.inter.gammas/spin('19F');

% Gamma-weighted detection states
coilPF6X=sparse(0); coilPF6Z=sparse(0);
for n=1:7
    coilPF6X=coilPF6X+weights(n)*(state(spin_system,{'L+'},{n})+state(spin_system,{'L-'},{n}))/2;
    coilPF6Z=coilPF6Z+weights(n)*state(spin_system,{'Lz'},{n});
end

% Gamma-weighted pulse operators
Sy=sparse(0); Sx=sparse(0);
for n=1:spin_system.comp.nspins
    Sy=Sy+weights(n)*(operator(spin_system,{'L+'},{n})-operator(spin_system,{'L-'},{n}))/2i;
    Sx=Sx+weights(n)*(operator(spin_system,{'L+'},{n})+operator(spin_system,{'L-'},{n}))/2;
end
timer.stop();
data.coil_x=full(coilPF6X); data.coil_z=full(coilPF6Z);

timer.start('pulse');
rho_pulse=evolution(spin_system,H+spin('19F')*30*1e-6*Sy,[],rho,1e-6,150,'final');
timer.stop();
data.rho_pulse=full(rho_pulse);

timer.start('acquisition');
fid=evolution(spin_system,H,coilPF6Z,rho_pulse,1/parameters.sweep,parameters.npoints-1,'observable');
timer.stop();
data.fid=fid;

timer.start('processing');

% Apodisation (legacy apodization(fid-mean(fid),'exp-1d',1))
fid_ap=apodisation(spin_system,fid-mean(fid),{{'exp',1}});

% Fourier transform
spectrum=fftshift(fft(fid_ap,parameters.zerofill));
timer.stop();
data.spectrum=spectrum;

% Plotting
timer.start('plot');
figs.spectrum=figure('Visible','off');
plot_1d(spin_system,abs(spectrum),parameters);
timer.stop();

end
