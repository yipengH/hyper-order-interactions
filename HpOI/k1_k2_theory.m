%% ==================================================================
% File: compute_phase_diagram.m
% Description:
%   This script computes the dynamical phase diagram of a mean-field
%   oscillator system with HpOI.
%
%   It scans the (K1, K2) parameter space and classifies the steady-state
%   dynamics into three regimes:
%       0 - Asynchronous state (no stable solution)
%       1 - Synchronous state (only stable solution exists)
%       2 - Bistable region (both stable and unstable solutions)
%
%   The classification is based on the implicit mean-field equation
%   and its derivative with respect to the order parameter r.
%
% Author: [Your Name]
% Date: [YYYY-MM-DD]
%% ==================================================================

%% Parameter setup
K1_values = -3:0.01:5;     % Range of first-order coupling strength
K2_values = 0:0.01:8;      % Range of second-order (HOI) coupling
sigma = 0.3;               % Adaptive exponent base (sigma = ε)

% Initialize matrix to store state classification
status_map = zeros(length(K1_values), length(K2_values));

% Define the derivative of the implicit equation dK1/dr
dK1_dr_expr = @(r, K1, K2, sigma) ...
    -2 * r * (0.5 * K1 * r + 0.5 * K2 * r^3 * sigma^(2 * r - 1)) + ...
    (1 - r^2) * (0.5 * K1 + K2 * r^3 * sigma^(2 * r - 1) * log(sigma) + ...
    1.5 * K2 * r^2 * sigma^(2 * r - 1)) - 1;

%% Parallel loop over K1 and K2 values
parfor i = 1:length(K1_values)
    K1 = K1_values(i);
    local_status = zeros(1, length(K2_values));  % Local result array

    for j = 1:length(K2_values)
        K2 = K2_values(j);
        stable_flag = 0;
        unstable_flag = 0;

        % Search for roots of the implicit equation for different initial r
        for r0 = linspace(0.01, 0.99, 200)
            f = @(r) -r + 0.5 * (K1 * r + sigma^(2*r - 1) * K2 * r^3) * (1 - r^2);
            try
                r_sol = fzero(f, r0);  % Attempt to find solution
                if abs(r_sol) < 1e-6  % Ignore near-zero roots
                    continue;
                end

                dr_sol = dK1_dr_expr(r_sol, K1, K2, sigma);
                if dr_sol > 0
                    unstable_flag = 1;
                else
                    stable_flag = 1;
                end
            catch
                continue;  % fzero failed, skip this trial
            end
        end

        % Classify state based on presence of stable/unstable roots
        if stable_flag == 0
            local_status(j) = 0;  % Asynchronous
        elseif unstable_flag == 0
            local_status(j) = 1;  % Fully synchronous
        else
            local_status(j) = 2;  % Bistable
        end
    end

    status_map(i, :) = local_status;  % Store results
end
