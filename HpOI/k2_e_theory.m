clear; clc; close all;

%% ============================================
% Description:
%   This script computes the dynamical state map of a coupled oscillator system
%   in the (K2, sigma) parameter space, at a fixed K1 value.
%   For each (K2, sigma) pair, it solves the fixed-point equation of the order parameter r,
%   and evaluates its stability using an analytical derivative expression.
%
%   States are classified as:
%     0 — Asynchronous (no stable solution)
%     1 — Synchronous (only stable solution(s))
%     2 — Bistable (both stable and unstable solutions coexist)
%
%   Additionally, the script identifies the closest pair of stable and unstable solutions
%   in terms of r-value difference.
% Date: May 2025
%% ============================================

% Parameter setup
K2_values = 0:0.01:8;                         % Range of K2 values
sigma_values = logspace(-1, 1, 801);          % Range of sigma values (log scale)
K1 = 0.1;                                     % Fixed K1 value

% Result matrix to store classification of each (K2, sigma) pair
status_map = zeros(length(K2_values), length(sigma_values));

% Implicit derivative expression dK1/dr used to assess stability
dK1_dr_expr = @(r, K1, K2, sigma) ...
    -2 * r * (0.5 * K1 * r + 0.5 * K2 * r^3 * sigma^(2 * r - 1)) + ...
    (1 - r^2) * (0.5 * K1 + K2 * r^3 * sigma^(2 * r - 1) * log(sigma) + ...
    1.5 * K2 * r^2 * sigma^(2 * r - 1)) - 1;


% Scan the (K2, sigma) parameter space
for i = 1:length(K2_values)
    for j = 1:length(sigma_values)
        K2 = K2_values(i);
        sigma = sigma_values(j);

        stable_solutions = [];
        unstable_solutions = [];

        % Use multiple initial guesses to find all possible roots
        for r0 = linspace(0.01, 0.99, 10)
            f = @(r) -r + 0.5 * (K1 * r + sigma.^(2*r - 1) * K2 * r.^3) .* (1 - r.^2);
            try
                r_sol = fzero(f, r0);
                if r_sol < 1e-6  % Ignore near-zero solutions
                    continue;
                end

                % Check the stability of the solution
                dr_sol = dK1_dr_expr(r_sol, K1, K2, sigma);
                if dr_sol > 0
                    unstable_solutions = [unstable_solutions, r_sol];
                else
                    stable_solutions = [stable_solutions, r_sol];
                end
            catch
                continue;
            end
        end

        % Classify state based on number and type of solutions
        if isempty(stable_solutions)
            status_map(i, j) = 0;  % Asynchronous
        elseif isempty(unstable_solutions)
            status_map(i, j) = 1;  % Synchronous
        else
            status_map(i, j) = 2;  % Bistable
        end


    end
end
