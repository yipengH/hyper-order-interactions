clc; clear; close all;
rng(7)  % Set random seed for reproducibility

%% ================================================
% Objective: Explore synchronization (order parameter r) as J1 varies,
%            both via forward/backward numerical integration and theory.
%
% Forward/backward simulations reveal hysteresis behavior.
% Theoretical solutions derived via self-consistency equation.
% =================================================

%% Parameters
N_nod = 100000;             % Number of oscillators
betax = 0; betay = 0;       % Phase lag (not used in this version)
epsilon = 3;                % Adaptive control parameter
delta = 1;                  % Unused here
T_sim = 20; T0 = 100;       % Simulation + transient time
T_tot = T_sim + T0;
step = 0.01;                % Time step
Time = round(T_tot / step); % Total time steps

% Natural frequency distribution (Lorentzian)
w0 = 0;
w_x = Lorentzian_distribution(w0, N_nod);  % Natural frequencies
x = pi * (-1 + 2 * rand(1, N_nod));        % Initial phases: uniform in [-pi, pi]

% Transfer to GPU for acceleration
w_x = gpuArray(w_x); 
x = gpuArray(x); 
x00 = x;                     % Save initial condition for backward scan
tt = 100;                    % Sampling interval (in steps)
J1 = -3:0.2:5;               % Coupling strength scan (J1)
J2 = 4;                      % Fixed second-order coupling

% Containers
r = zeros(Time / tt, 1);     % Instantaneous order parameter
r_fw = zeros(size(J1));      % Forward bifurcation
r_bw = zeros(size(J1));      % Backward bifurcation

%% Forward Simulation
for i = 1:length(J1)
    nn = 1;
    x0 = x;  % reset to previous x
    for t = 1:Time
        % Adaptive coupling adjustment
        x1 = mod(x, 2*pi) - pi;
        adaptive = abs(mean(exp(1i * x1)));
        adaptive_x = epsilon^(2 * adaptive - 1);

        % Compute force
        F_x = funx_force(x, J1(i), adaptive_x * J2, betax);

        % Euler integration
        x = x0 + step * (w_x + F_x);
        x0 = x;

        % Record average order parameter
        if t >= T0 && mod(t, tt) == 0
            re = mean(cos(x)); im = mean(sin(x));
            r(nn) = sqrt(re^2 + im^2);
            t_all(nn) = t / tt;
            nn = nn + 1;
        end
    end
    r_fw(i) = mean(r(round(T_sim / (step * tt)):end));  % Average in steady state
end

%% Backward Simulation (starting from last state)
x0 = x00;
for i = length(J1):-1:1
    nn = 1;
    x0 = x;
    for t = 1:Time
        x1 = mod(x, 2*pi) - pi;
        adaptive = abs(mean(exp(1i * x1)));
        adaptive_x = epsilon^(2 * adaptive - 1);
        F_x = funx_force(x, J1(i), adaptive_x * J2, betax);
        x = x0 + step * (w_x + F_x);
        x0 = x;

        if t >= T0 && mod(t, tt) == 0
            re = mean(cos(x)); im = mean(sin(x));
            r(nn) = sqrt(re^2 + im^2);
            t_all(nn) = t / tt;
            nn = nn + 1;
        end
    end
    r_bw(i) = mean(r(round(T_sim / (step * tt)):end));
end

%% Theoretical Solutions (Self-consistent Equation)
J1_values = linspace(-3, 5, 500);
r_init = linspace(0.01, 0.99, 200);  % Initial guesses

r_solutions = []; dr_solutions = []; K1_solutions = [];

% Derivative expression for stability test
dK1_dr_expr = @(r, J1, J2, sigma) ...
    -2 * r * (0.5 * J1 * r + 0.5 * J2 * r^3 * sigma^(2 * r - 1)) + ...
    (1 - r^2) * (0.5 * J1 + J2 * r^3 * sigma^(2 * r - 1) * log(sigma) + ...
    1.5 * J2 * r^2 * sigma^(2 * r - 1)) - 1;

for J1 = J1_values
    for r0 = r_init
        f = @(r) -r + 0.5 * (J1 * r + epsilon.^(2*r - 1) * J2 * r.^3) .* (1 - r.^2);
        try
            r_sol = fzero(f, r0);
            dr_sol = dK1_dr_expr(r_sol, J1, J2, epsilon);
            if r_sol > 0 && r_sol < 1 && all(abs(r_sol - r_solutions) > 1e-3)
                r_solutions(end+1) = r_sol;
                dr_solutions(end+1) = dr_sol;
                K1_solutions(end+1) = J1;
            end
        catch
            continue;
        end
    end
end

%% Plotting Bifurcation Diagram
% Separate stable and unstable solutions
unstable_r_solutions = [];
stable_r_solutions = [];
unstable_K1_solutions = [];
stable_K1_solutions = [];

for i = 1:length(dr_solutions)
    if dr_solutions(i) > 0
        unstable_r_solutions(end+1) = r_solutions(i);
        unstable_K1_solutions(end+1) = K1_solutions(i);
    else
        stable_r_solutions(end+1) = r_solutions(i);
        stable_K1_solutions(end+1) = K1_solutions(i);
    end
end

% Filter stable states (e.g., avoid near-zero values)
valid_idx = stable_r_solutions >= 0.2;
stable_r_solutions_filtered = stable_r_solutions(valid_idx);
stable_K1_solutions_filtered = stable_K1_solutions(valid_idx);

% Plotting
figure; hold on; box on;
plot(stable_K1_solutions_filtered, stable_r_solutions_filtered, 'b-', 'LineWidth', 2, 'DisplayName', 'Stable solution');
plot(unstable_K1_solutions, unstable_r_solutions, 'r--', 'LineWidth', 2, 'DisplayName', 'Unstable solution');
scatter(J1, r_fw, 60, 'b', 'filled', 'v', 'DisplayName', 'Forward simulation');
scatter(J1, r_bw, 60, 'k', 'filled', '^', 'DisplayName', 'Backward simulation');

xlabel('J_1'); ylabel('r');
ylim([0 1]); xlim([min(J1_values), max(J1_values)]);
legend('Location', 'southeast');
title('Theoretical and Numerical Bifurcation Diagram');
set(gca, 'FontSize', 12);
hold off;

%% Function Definitions

% Compute coupling force based on Kuramoto model with second harmonics
function [F_x] = funx_force(x, J1, J2, beta)
    z1 = mean(exp(1i * x));
    z2 = mean(exp(2i * x));
    F_x = J1 * imag(z1 .* exp(-1i * (x + beta))) + ...
          J2 * imag(z2 .* conj(z1) .* exp(-1i * (x + beta)));
end

% Sample from Lorentzian distribution centered at w0
function [w] = Lorentzian_distribution(w0, num_samples)
    pdf = @(w) 1 ./ (pi * (1 + (w - w0).^2));
    w_values = linspace(-200, 200, num_samples);
    cdf_values = cumtrapz(w_values, pdf(w_values));
    cdf_values = cdf_values / cdf_values(end);  % Normalize to [0, 1]
    inverse_cdf = griddedInterpolant(cdf_values, w_values, 'linear');
    uniform_random_numbers = rand(num_samples, 1);
    w = inverse_cdf(uniform_random_numbers)';
end
