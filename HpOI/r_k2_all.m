clc; clear; close all;
rng(7);  % Set random seed

%% ===============================================================
% Description:
%   Simulates synchronization dynamics in a system of N oscillators with
%   HpOI. The forward and backward bifurcation
%   curves are compared against theoretical predictions by solving implicit
%   mean-field equations.
%
% Model:
%   - Phase oscillators with Lorentzian-distributed natural frequencies
%   - Adaptive coupling strength: sigma = epsilon^(2r - 1)
%   - Hysteresis generated via forward/backward scan over J2 (HOI strength)
%
% Outputs:
%   - r_fw: forward order parameter curve
%   - r_bw: backward order parameter curve
%   - Bifurcation diagram overlaying theory and simulation

%% ===============================================================

%% Initialization
la = load('.\x_ini.mat');  % Load synchronized state for backward scan
N_nod = 100000;
betax = 0; betay = betax;
epsilon = 0.1; delta = 1;

T_sim = 20; T0 = 100; T_tot = T_sim + T0;
step = 0.01; Time = round(T_tot / step);
tt = 100;  % Sampling interval
w0 = 0;

% Generate Lorentzian-distributed natural frequencies
[w_x] = Lorentzian_distribution(w0, N_nod);
x = pi * (-1 + 2 * rand(1, N_nod));  % Initial phases: uniform [-pi, pi]
w_x = gpuArray(w_x); x = gpuArray(x); x00 = x;
x_sync = gpuArray(la.x);

J1 = 1; J2 = 0:0.2:8;  % Coupling strengths to sweep
r = zeros(Time/tt, 1);

%% Forward scan over J2
for i = 1:length(J2)
    nn = 1; x0 = x;
    fprintf('Forward: J2 = %.2f\n', J2(i));
    for t = 1:Time
        adaptive = abs(mean(exp(1i * x)));
        adaptive_x = epsilon^(2 * adaptive - 1);
        F_x = funx_force(x, J1, adaptive_x * J2(i), betax);
        x = x0 + step * (w_x + F_x);
        x0 = x;
        if t >= T0 && mod(t, tt) == 0
            re = mean(cos(x)); im = mean(sin(x));
            r(nn) = sqrt(re^2 + im^2);
            nn = nn + 1;
        end
    end
    r_fw(i) = mean(r(round(T_sim / (step * tt)):end));
end

%% Backward scan from synchronized initial state
x = x_sync;
for i = length(J2):-1:1
    nn = 1; x0 = x;
    fprintf('Backward: J2 = %.2f\n', J2(i));
    for t = 1:Time
        adaptive = abs(mean(exp(1i * x)));
        adaptive_x = epsilon^(2 * adaptive - 1);
        F_x = funx_force(x, J1, adaptive_x * J2(i), betax);
        x = x0 + step * (w_x + F_x);
        x0 = x;
        if t >= T0 && mod(t, tt) == 0
            re = mean(cos(x)); im = mean(sin(x));
            r(nn) = sqrt(re^2 + im^2);
            nn = nn + 1;
        end
    end
    r_bw(i) = mean(r(round(T_sim / (step * tt)):end));
end

%% Theoretical bifurcation analysis
J2_values = linspace(0, 8, 500);
r_init = linspace(0.01, 0.99, 200);

r_solutions = []; dr_solutions = []; K2_solutions = [];

% Derivative of J1 with respect to r from implicit equation
dK1_dr_expr = @(r, J1, J2, sigma) -2 * r * (0.5 * J1 * r + 0.5 * J2 * r^3 * sigma^(2 * r - 1)) + ...
    (1 - r^2) * (0.5 * J1 + J2 * r^3 * sigma^(2 * r - 1) * log(sigma) + ...
    1.5 * J2 * r^2 * sigma^(2 * r - 1)) - 1;

for J2 = J2_values
    for r0 = r_init
        f = @(r) -r + 0.5 * (J1 * r + epsilon^(2*r - 1) * J2 * r^3) * (1 - r^2);
        try
            r_sol = fzero(f, r0);
            dr_sol = dK1_dr_expr(r_sol, J1, J2, epsilon);
            if r_sol > 0 && r_sol < 1 && all(abs(r_sol - r_solutions) > 1e-3)
                r_solutions(end+1) = r_sol;
                dr_solutions(end+1) = dr_sol;
                K2_solutions(end+1) = J2;
            end
        catch
            continue;
        end
    end
end

%% Plotting: Bifurcation Diagram
unstable_r_solutions = []; stable_r_solutions = [];
unstable_K2_solutions = []; stable_K2_solutions = [];

for i = 1:length(dr_solutions)
    if dr_solutions(i) > 0
        unstable_r_solutions(end+1) = r_solutions(i);
        unstable_K2_solutions(end+1) = K2_solutions(i);
    else
        stable_r_solutions(end+1) = r_solutions(i);
        stable_K2_solutions(end+1) = K2_solutions(i);
    end
end

% Filter stable r values
valid_idx = stable_r_solutions >= 0.2;
stable_r_filtered = stable_r_solutions(valid_idx);
stable_K2_filtered = stable_K2_solutions(valid_idx);
J2_sim = 0:0.2:8;

figure; hold on; box on;

% Plot theoretical branches
plot(stable_K2_filtered, stable_r_filtered, 'b-', 'LineWidth', 2, 'DisplayName', 'Stable solution');
plot(unstable_K2_solutions, unstable_r_solutions, 'r--', 'LineWidth', 2, 'DisplayName', 'Unstable solution');

% Plot simulation results
scatter(J2_sim, r_fw, 60, 'b', 'filled', 'v', 'DisplayName', 'Forward simulation');
scatter(J2_sim, r_bw, 60, 'k', 'filled', '^', 'DisplayName', 'Backward simulation');

xlabel('J_2'); ylabel('r');
ylim([0 1]); xlim([min(J2_values), max(J2_values)]);
legend('Location', 'southeast');
title('Theoretical and Numerical Bifurcation Diagram');
set(gca, 'FontSize', 12);
hold off;

%% ---------------------------------------------------------------
% Function: Interaction force from first and second harmonics
function [F_x] = funx_force(x, J1, J2, beta)
z1 = mean(exp(1i * x));
z2 = mean(exp(2i * x));
F_x = J1/2 * (2 * imag(z1 .* exp(-1i * (x + beta)))) + ...
      J2/2 * (2 * imag(z2 .* conj(z1) .* exp(-1i * (x + beta))));
end

%% Function: Generate Lorentzian-distributed frequencies
function [w] = Lorentzian_distribution(w0, num_samples)
pdf = @(w) 1 ./ (pi * (1 + (w - w0).^2));
w_values = linspace(-200, 200, num_samples);
cdf_values = cumtrapz(w_values, pdf(w_values));
cdf_values = cdf_values / cdf_values(end);
inverse_cdf = griddedInterpolant(cdf_values, w_values, 'linear');
w = inverse_cdf(rand(num_samples, 1))';
end
