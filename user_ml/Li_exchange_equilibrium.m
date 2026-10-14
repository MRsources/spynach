% Issue with setting the proper equilibrium density operator for 
% 7Li in two-site chemical exchange
%
% Florin, 02/01/2026
clear all; close all; 

%% --- Magnet & spins ---
sys.magnet   = 1.93;  % nu_7Li=32MHz                  
sys.isotopes = {'7Li','7Li'};      

%% --- Quadrupolar tensors for A and B (Cq in Hz, eta, I=3/2, Euler[rad]) ---
Cq=30*1e3; 
inter.coupling.matrix{1,1} = eeqq2nqi(Cq, 0, 3/2, [0 0 0]);
inter.coupling.matrix{2,2} = eeqq2nqi(3*Cq, 0, 3/2, [0 0 0]);

%% --- Chemical shifts (ppm) ---
sigma_iso=-1; delta_sigma=45; eta=0;
inter.zeeman.matrix{1} = [sigma_iso-delta_sigma/2*(1+eta) 0 0; ...
                          0 sigma_iso-delta_sigma/2*(1-eta) 0; ...
                          0 0 sigma_iso+delta_sigma];
inter.zeeman.matrix{2} = [sigma_iso-3*delta_sigma/2*(1+eta) 0 0; ...
                          0 sigma_iso-3*delta_sigma/2*(1-eta) 0; ...
                          0 0 sigma_iso+3*delta_sigma];

%% --- Chemical exchange: A <-> B ---
inter.chem.parts = {1,2}; 
inter.temperature  = 298;
pA=0.95;pB=1-pA;
kex=1e3;
kAB = pB*kex;                       
kBA = pA*kex;                         
inter.chem.rates = [-kAB  kBA; 
                     kAB -kBA];
% Initial concentrations
inter.chem.concs = [1 kAB/kBA];   
inter.chem.concs=equilibrate(inter.chem.rates,inter.chem.concs'); %sanity check

%% --- Relaxation model ---
inter.relaxation   = {'redfield'};
inter.rlx_keep     = 'secular';
inter.equilibrium  = 'zero';   
inter.tau_c        = {1e-9,1e-9};   

%% --- Basis & formalism ---
bas.formalism     = 'sphten-liouv';
bas.approximation = 'none';

%% --- Spinach init ---
spin_system = create(sys, inter);
spin_system = basis(spin_system, bas);

%% --- Liouvillian & equilibrium state ---
H0     = hamiltonian(assume(spin_system,'labframe'),'left');
rho_eq = equilibrium(spin_system,H0);

Lx=operator(spin_system,'Lx','7Li');
H=hamiltonian(assume(spin_system,'nmr'));
K=kinetics(spin_system);
rho_eq_ex=evolution(spin_system,H+1i*K,[],rho_eq,1e-5,1e6,'final'); %10s for 'chemical' equilibration

% R=relaxation(spin_system);
R=thermalize(spin_system,relaxation(spin_system),[],[],rho_eq_ex,'IME');
L0=H+1i*R+1i*K;
rho_pert=step(spin_system,Lx,rho_eq_ex,0); %apply perturbation
traj_eq=evolution(spin_system,L0,[],rho_pert,1e-5,2*1e5,'trajectory'); %2s for full equilibration

%% Analyze trajectory during equilibration
figure(1); hold on;
time_eq=linspace(0,1,size(traj_eq,2));

% Display the evolution of Tl,0 tensors
T10=state(spin_system,'T1,0','7Li','chem');
T20=state(spin_system,'T2,0','7Li','chem');
T30=state(spin_system,'T3,0','7Li','chem');
subplot(1,4,1); hold on;
plot(time_eq,T10'*traj_eq,time_eq,T20'*traj_eq,time_eq,T30'*traj_eq,'LineWidth',5);
legend('T10','T20','T30');
xlabel('Equilibration time (s)');

% Display the evolution of states with different correlation orders
subplot(1,4,2); hold on;
trajan(spin_system,traj_eq,'correlation_order',time_eq);

% Display the evolution of states with different coherence orders
subplot(1,4,3); hold on;
trajan(spin_system,traj_eq,'coherence_order',time_eq);

% Display the evolution of total population on different spins
subplot(1,4,4); hold on;
trajan(spin_system,traj_eq,'total_each_spin',time_eq);

%% Density operator analysis
disp('Density Operator after equilibrium(H0)');
stateinfo(spin_system,rho_eq,16);

disp('Density Operator after H+1i*K (targeted state)');
stateinfo(spin_system,rho_eq_ex,16);

disp('Density Operator after perturbation + H+1i*R+1i*K');
stateinfo(spin_system,traj_eq(:,end),16);

disp(['Unity State population in the targeted state: ',num2str(state(spin_system,'E','all')'*rho_eq_ex/norm(state(spin_system,'E','all')))])