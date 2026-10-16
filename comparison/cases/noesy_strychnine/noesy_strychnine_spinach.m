% Comparison case: NOESY of strychnine, examples/nmr_liquids/noesy_strychnine.m
% with timed stages. Stage labels must match noesy_strychnine_spynach.py.
% Syntax:
%
%     [data,figs]=noesy_strychnine_spinach(timer,options)
%
% Parameters:
%
%     timer   - cmp_timer object (timer.start(label) ... timer.stop())
%
%     options - struct, may be empty; fields npoints (default 512)
%               and zerofill (default 2048) apply to both dimensions;
%               gpu (default false) adds 'gpu' to sys.enable
%
% Outputs:
%
%     data    - struct of arrays saved to data.mat: fid_cos, fid_sin
%               (raw FIDs, before apodisation) and spectrum
%
%     figs    - struct of figure handles saved as figure_<name>.png

function [data,figs]=noesy_strychnine_spinach(timer,options)

if nargin<2, options=struct(); end
npoints=cmp_option(options,'npoints',512);
zerofill=cmp_option(options,'zerofill',2048);

% Spin system properties
[sys,inter]=strychnine({'1H'});

% Magnet field
sys.magnet=5.9;

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='IK-2';
bas.connectivity='scalar_couplings';
bas.prox_level=3;

% Relaxation theory parameters
inter.relaxation={'redfield'};
inter.equilibrium='IME';
inter.temperature=298;
inter.rlx_keep='kite';
inter.tau_c={200e-12};

% Algorithmic options
sys.enable={'greedy'};
if cmp_option(options,'gpu',false), sys.enable=[sys.enable {'gpu'}]; end
sys.disable={'krylov'};
sys.tols.prox_cutoff=4.0;

% Spinach housekeeping
timer.start('create');
spin_system=create(sys,inter);
timer.stop();
timer.start('basis');
spin_system=basis(spin_system,bas);
timer.stop();

% Sequence parameters
parameters.tmix=0.5;
parameters.offset=1200;
parameters.sweep=[2500 2500];
parameters.npoints=[npoints npoints];
parameters.zerofill=[zerofill zerofill];
parameters.spins={'1H'};
parameters.axis_units='ppm';
parameters.needs={'rho_eq'};

% Simulation
timer.start('liquid/noesy');
fid=liquid(spin_system,@noesy,parameters,'nmr');
timer.stop();
data.fid_cos=fid.cos;
data.fid_sin=fid.sin;

timer.start('processing');

% Apodisation
fid.cos=apodisation(spin_system,fid.cos,{{'sqcos'},{'sqcos'}});
fid.sin=apodisation(spin_system,fid.sin,{{'sqcos'},{'sqcos'}});

% F2 Fourier transform
f1_cos=real(fftshift(fft(fid.cos,parameters.zerofill(2),1),1));
f1_sin=real(fftshift(fft(fid.sin,parameters.zerofill(2),1),1));

% States signal
f1_states=f1_cos-1i*f1_sin;

% F1 Fourier transform
spectrum=fftshift(fft(f1_states,parameters.zerofill(1),2),2);
timer.stop();
data.spectrum=spectrum;

% Plotting
timer.start('plot');
figs.noesy=figure('Visible','off'); scale_figure([1.5 2.0]);
plot_2d(spin_system,-real(spectrum),parameters,...
        20,[0.01 0.1 0.01 0.1],2,256,6,'both');
timer.stop();

end
