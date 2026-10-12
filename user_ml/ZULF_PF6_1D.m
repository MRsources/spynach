% Saturation Transfer ZULF with selective transition pulse
% ZULF Z-spectrum of PF6
%
% Florin 01/02/2024
clear all;
close all; 

% Solenoid Field
sys.magnet=2.7*1e-6;
sys.output='hush';
sys.enable={'greedy'};

% Spin system
sys.isotopes={'31P','19F','19F','19F','19F','19F','19F'};       

% 1J_PF coupling system for 2M LiPF6
J_PF=710; rPF=1.6;
inter.coupling.scalar{1,2}=J_PF; %1J P-F
inter.coupling.scalar{1,3}=J_PF; %1J P-F
inter.coupling.scalar{1,4}=J_PF; %1J P-F
inter.coupling.scalar{1,5}=J_PF; %1J P-F
inter.coupling.scalar{1,6}=J_PF; %1J P-F
inter.coupling.scalar{1,7}=J_PF; %1J P-F
inter.coupling.scalar{7,7}=0;
inter.coordinates=rPF*{[0 0 0],[-1 0 0],[1 0 0],[0 -1 0],[0 1 0],[0 0 -1],[0 0 1]};

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='none';

% Relaxation theory
inter.relaxation={'redfield'};
inter.equilibrium='dibari';
inter.temperature=298;
inter.rlx_keep='labframe';
inter.tau_c={0.065e-9}; % predicted from experimental data (see PF6_EC_DMC_tau_C_prediction.nb)

% Spinach housekeeping
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);
H=hamiltonian(assume(spin_system,'labframe'),'comm');

% Read density operator after adiabatic drop in Anne's ZULF BELT
rho=csvread("/Users/teleanuflorin/Documents/Alexej_phosphate/ZULF_Sat_transfer/Notebooks/PF6_battery/rho_adiabatic_drop.csv");

% Sequence parameters
parameters.sweep=6000;% Dwell time used by Anne
parameters.npoints=4*4096;
parameters.zerofill=4*4096; %to match exp spectrum
parameters.offset=0;
parameters.spins={'19F'};
parameters.axis_units='Hz';
parameters.invert_axis=0;
parameters.detection='quadrature';

% Magnetogyric ratio weights relative to 19F
weights=spin_system.inter.gammas/spin('19F');

% Get gamma-weighted detection state
coilPF6X=sparse(0); coilPF6Z=sparse(0);
for n=1:7
    coilPF6X=coilPF6X+weights(n)*(state(spin_system,{'L+'},{n})+state(spin_system,{'L-'},{n}))/2;
    coilPF6Z=coilPF6Z+weights(n)*state(spin_system,{'Lz'},{n});
end

% Get gamma-weighted pulse operator
Sy=sparse(0);Sx=sparse(0);
for n=1:spin_system.comp.nspins
    Sy=Sy+weights(n)*(operator(spin_system,{'L+'},{n})-...
                      operator(spin_system,{'L-'},{n}))/2i;
    Sx=Sx+weights(n)*(operator(spin_system,{'L+'},{n})+...
                      operator(spin_system,{'L-'},{n}))/2;
end

rho_pulse=evolution(spin_system,H+spin('19F')*30*1e-6*Sy,[],rho,1e-6,150,'final');
fid=evolution(spin_system,H,coilPF6Z,rho_pulse,1/parameters.sweep,parameters.npoints-1,'observable');
% Apodization
fid_ap=apodization(fid-mean(fid),'exp-1d',1);

% Fourier transform
spectrum=fftshift(fft(fid_ap,parameters.zerofill));

figure(1); hold on;
plot_1d(spin_system,abs(spectrum),parameters);
%xlim([80 140]);
% % Run frequency scan saturation
% freq_list=linspace(1,7,7); max_spec=[];
% for k=1:size(freq_list,2)
% npoints=1000;
% duration=1; %s 
% power=500; %
% freq=repelem(0.5*inter.coupling.scalar{1,2}*freq_list(k),npoints);
% dur=duration*ones(1,npoints)/npoints;
% [amp,phase]=read_wave('rectangular_1000.pk',npoints);
% amp=2*pi*power*amp;
% 
% rho1=shaped_pulse_af(spin_system,H,Sx,Sy,rho,freq,amp,dur,phase(1),2);
% 
% % Sequence parameters
% parameters.sweep=700;
% parameters.npoints=4096;
% parameters.zerofill=8192;
% parameters.offset=0;
% parameters.axis_units='Hz';
% parameters.spins={'1H'};
% 
% % Apply the pulse
% rho_pulse=evolution(spin_system,H+spin('19F')*30*1e-6*Sy,[],rho,1e-6,600,'trajectory');
% 
% fid=evolution(spin_system,H+1i*R,coil,rho_pulse,1/parameters.sweep,parameters.npoints-1,'observable');
% 
% % Apodization
% fid=apodization(fid-mean(fid),'exp-1d',6);
% 
% % Fourier transform
% spectrum=fftshift(fft(fid,parameters.zerofill));
% 
% max_spec(k)=max(abs(spectrum));
% 
% end
% 
% % Plot ZULF Z-spectrum 
% parameters.invert_axis=0;
% parameters.flip_angle=pi/2;
% parameters.detection='uniaxial';
% fid=liquid(spin_system,@zerofield,parameters,'labframe');
% fid=fid*exp(1i*pi/2);
% 
% % Apodization
% fid=apodization(fid-mean(fid),'exp-1d',6);
% 
% % Fourier transform
% spectrum=fftshift(fft(fid,parameters.zerofill));
% 
% figure(1); hold on;
% subplot(1,2,1); hold on;
% plot(freq_list*inter.coupling.scalar{1,2},max_spec/max(max_spec))
% xlabel('Frequency offset [Hz]'); ylabel('ZULF Z-spectrum');
% legend('Power 20 Hz','Power 40 Hz');
% xlim([0 350]); ylim([0 1]);
% 
% subplot(1,2,2); hold on; 
% plot_1d(spin_system,abs(spectrum)/max(abs(spectrum)),parameters);
% xlim([0 350]); ylim([0 1]);
% 
% % Theoretical frequencies in ULF spectrum (Appelt et al 2006)
% function energy = energy2(F, FI, mf,J, B)
% 
%     energy = mf * spin('1H')/(2*pi) * B - ...
%              J/4 + ...
%              (F - FI) * ((FI + 1/2)^2 * J^2 +2 * mf * J * (spin('13C')/(2*pi) - spin('1H')/(2*pi)) * B + (spin('13C')/(2*pi) - spin('1H')/(2* pi))^2 * B^2)^0.5;
% end