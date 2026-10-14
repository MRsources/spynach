% Comparison case: equilibration of a two-site 7Li exchange system,
% user_ml/Li_exchange_equilibrium.m (Florin Teleanu, 2026) with timed
% stages. The four subplots of the script are saved as four figures and
% the trajectory analysis of trajan is also returned as arrays. Stage
% labels must match li_exchange_equilibrium_respinach.py. Syntax:
%
%     [data,figs]=li_exchange_equilibrium_spinach(timer,options)
%
% Parameters:
%
%     timer   - cmp_timer object
%
%     options - struct, may be empty; nsteps_ex (default 1e6, steps of
%               10 us of chemical equilibration under H+1i*K) and
%               nsteps_eq (default 2e5, steps of the trajectory);
%               gpu (default false) adds 'gpu' to sys.enable
%
% Outputs:
%
%     data    - rho_eq, rho_eq_ex, rho_final, tensors (T10, T20, T30
%               projections along the trajectory), corr_order,
%               coh_order, spin_total (trajan amplitudes),
%               unit_population, time_eq
%
%     figs    - tensors, correlation, coherence, spins

function [data,figs]=li_exchange_equilibrium_spinach(timer,options)

if nargin<2, options=struct(); end
nsteps_ex=cmp_option(options,'nsteps_ex',1e6);
nsteps_eq=cmp_option(options,'nsteps_eq',2e5);

% Magnet and spins
sys.magnet=1.93;  % nu_7Li=32MHz
sys.isotopes={'7Li','7Li'};
if cmp_option(options,'gpu',false), sys.enable={'gpu'}; end

% Quadrupolar tensors for A and B (Cq in Hz, eta, I=3/2, Euler[rad])
Cq=30*1e3;
inter.coupling.matrix{1,1}=eeqq2nqi(Cq,0,3/2,[0 0 0]);
inter.coupling.matrix{2,2}=eeqq2nqi(3*Cq,0,3/2,[0 0 0]);

% Chemical shifts (ppm)
sigma_iso=-1; delta_sigma=45; eta=0;
inter.zeeman.matrix{1}=[sigma_iso-delta_sigma/2*(1+eta) 0 0; ...
                        0 sigma_iso-delta_sigma/2*(1-eta) 0; ...
                        0 0 sigma_iso+delta_sigma];
inter.zeeman.matrix{2}=[sigma_iso-3*delta_sigma/2*(1+eta) 0 0; ...
                        0 sigma_iso-3*delta_sigma/2*(1-eta) 0; ...
                        0 0 sigma_iso+3*delta_sigma];

% Chemical exchange: A <-> B
inter.chem.parts={1,2};
inter.temperature=298;
pA=0.95; pB=1-pA;
kex=1e3;
kAB=pB*kex;
kBA=pA*kex;
inter.chem.rates=[-kAB  kBA;
                   kAB -kBA];
inter.chem.concs=[1 kAB/kBA];
inter.chem.concs=equilibrate(inter.chem.rates,inter.chem.concs'); % sanity check

% Relaxation model
inter.relaxation={'redfield'};
inter.rlx_keep='secular';
inter.equilibrium='zero';
inter.tau_c={1e-9,1e-9};

% Basis and formalism
bas.formalism='sphten-liouv';
bas.approximation='none';

% Spinach init
timer.start('create');
spin_system=create(sys,inter);
timer.stop();
timer.start('basis');
spin_system=basis(spin_system,bas);
timer.stop();

% Liouvillian and equilibrium state
timer.start('hamiltonian');
H0=hamiltonian(assume(spin_system,'labframe'),'left');
Lx=operator(spin_system,'Lx','7Li');
H=hamiltonian(assume(spin_system,'nmr'));
K=kinetics(spin_system);
timer.stop();
timer.start('equilibrium');
rho_eq=equilibrium(spin_system,H0);
rho_eq_ex=evolution(spin_system,H+1i*K,[],rho_eq,1e-5,nsteps_ex,'final'); % 10s for 'chemical' equilibration
timer.stop();
timer.start('relaxation');
R=thermalize(spin_system,relaxation(spin_system),[],[],rho_eq_ex,'IME');
L0=H+1i*R+1i*K;
timer.stop();
timer.start('trajectory');
rho_pert=step(spin_system,Lx,rho_eq_ex,0); % apply perturbation
traj_eq=evolution(spin_system,L0,[],rho_pert,1e-5,nsteps_eq,'trajectory'); % 2s for full equilibration
timer.stop();

% Analyze trajectory during equilibration
timer.start('analysis');
time_eq=linspace(0,1,size(traj_eq,2));
T10=state(spin_system,'T1,0','7Li','chem');
T20=state(spin_system,'T2,0','7Li','chem');
T30=state(spin_system,'T3,0','7Li','chem');
data.tensors=[T10'*traj_eq; T20'*traj_eq; T30'*traj_eq];
data.corr_order=trajan_data(spin_system,traj_eq,'correlation_order');
data.coh_order=trajan_data(spin_system,traj_eq,'coherence_order');
data.spin_total=trajan_data(spin_system,traj_eq,'total_each_spin');

% Density operator analysis
disp('Density Operator after equilibrium(H0)');
stateinfo(spin_system,rho_eq,16);
disp('Density Operator after H+1i*K (targeted state)');
stateinfo(spin_system,rho_eq_ex,16);
disp('Density Operator after perturbation + H+1i*R+1i*K');
stateinfo(spin_system,traj_eq(:,end),16);
unit_population=state(spin_system,'E','all')'*rho_eq_ex/norm(state(spin_system,'E','all'));
disp(['Unity State population in the targeted state: ',num2str(unit_population)]);
timer.stop();
data.rho_eq=full(rho_eq); data.rho_eq_ex=full(rho_eq_ex); data.rho_final=full(traj_eq(:,end));
data.unit_population=unit_population; data.time_eq=time_eq;

% Plotting: the four subplots of the script as four figures
timer.start('plot');
figs.tensors=figure('Visible','off'); hold on;
plot(time_eq,data.tensors(1,:),time_eq,data.tensors(2,:),time_eq,data.tensors(3,:),'LineWidth',5);
legend('T10','T20','T30'); xlabel('Equilibration time (s)');
figs.correlation=figure('Visible','off'); hold on; trajan(spin_system,traj_eq,'correlation_order',time_eq);
figs.coherence=figure('Visible','off'); hold on; trajan(spin_system,traj_eq,'coherence_order',time_eq);
figs.spins=figure('Visible','off'); hold on; trajan(spin_system,traj_eq,'total_each_spin',time_eq);
timer.stop();

end

% The amplitudes that trajan.m plots, as an array with one row per class
function result=trajan_data(spin_system,traj,property)
unit=unit_state(spin_system); traj=traj-(unit*unit')*traj;
switch property
    case 'correlation_order'
        orders=sum(logical(spin_system.bas.basis),2);
        classes=setdiff(unique(orders),0);
        masks=arrayfun(@(o) orders==o,classes,'UniformOutput',false);
    case 'coherence_order'
        [~,M]=lin2lm(spin_system.bas.basis); orders=sum(M,2);
        classes=unique(orders);
        masks=arrayfun(@(o) orders==o,classes,'UniformOutput',false);
    case 'total_each_spin'
        masks=arrayfun(@(n) spin_system.bas.basis(:,n)~=0,1:spin_system.comp.nspins,'UniformOutput',false);
end
result=zeros(numel(masks),size(traj,2));
for n=1:numel(masks)
    subspace_trajectory=traj(masks{n},:);
    result(n,:)=sqrt(sum(subspace_trajectory.*conj(subspace_trajectory),1));
end
end
