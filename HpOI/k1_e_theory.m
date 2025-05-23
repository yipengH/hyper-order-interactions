clear; clc; close all;

%% ============================================
% Description:
%   This script computes the dynamical state map of a coupled oscillator model
%   in the (K1, K2) parameter space at a fixed sigma value.
%   It solves the fixed-point equation for r using multiple initial guesses,
%   and uses the derivative expression to classify the stability of solutions.
%
%   States are classified as:
%     0 — Asynchronous (no stable solution)
%     1 — Synchronous (only stable solution(s))
%     2 — Bistable (both stable and unstable solutions coexist)
%
% Parallel computing (`parfor`) is used to accelerate the evaluation.

% Date: May 2025
%% ============================================

% Parameter setup
K1_values = -3:0.01:5;          % Range of K1 values
K2_values = 0:0.01:8;           % Range of K2 values
sigma = 0.3;                    % Fixed sigma value

% Result matrix to store the classification of each (K1, K2) pair
status_map = zeros(length(K1_values), length(K2_values));

% Define the derivative expression dK1/dr used for stability analysis
dK1_dr_expr = @(r, K1, K2, sigma) ...
    -2 * r * (0.5 * K1 * r + 0.5 * K2 * r^3 * sigma^(2 * r - 1)) + ...
    (1 - r^2) * (0.5 * K1 + K2 * r^3 * sigma^(2 * r - 1) * log(sigma) + ...
    1.5 * K2 * r^2 * sigma^(2 * r - 1)) - 1;

% Main loop with parallel computing
parfor i = 1:length(K1_values)
    K1 = K1_values(i);
    local_status = zeros(1, length(K2_values));  % Local result vector for each K1

    for j = 1:length(K2_values)
        K2 = K2_values(j);
        stable_flag = 0;
        unstable_flag = 0;

        % Use multiple initial guesses to search for roots of the fixed-point equation
        for r0 = linspace(0.01, 0.99, 200)
            f = @(r) -r + 0.5 * (K1 * r + sigma.^(2 * r - 1) * K2 * r.^3) .* (1 - r.^2);
            try
                r_sol = fzero(f, r0);
                if abs(r_sol) < 1e-6
                    continue;  % Skip near-zero solutions
                end

                % Evaluate stability using the derivative
                dr_sol = dK1_dr_expr(r_sol, K1, K2, sigma);
                if dr_sol > 0
                    unstable_flag = 1;
                else
                    stable_flag = 1;
                end
            catch
                continue;  % Skip failed attempts
            end
        end

        % Classify the system state for the current (K1, K2)
        if stable_flag == 0
            local_status(j) = 0;  % Asynchronous
        elseif unstable_flag == 0
            local_status(j) = 1;  % Synchronous
        else
            local_status(j) = 2;  % Bistable
        end
    end

    % Save local results into the global map
    status_map(i, :) = local_status;
end
