clear; clc; close all;
rng(77);  % Set random seed for reproducibility

%% =============================================================
% File: simulate_adaptive_hoi_dynamics.m
% Description:
%   Simulate synchronization dynamics on a real-world network
%   (e.g., Dolphins network) with randomly constructed 3-node hyperedges
%   representing HpOI.
%   The model features adaptive coupling based on local triangle activity,
%   and performs both forward and backward bifurcation scans over J1.
%
% Steps:
%   1. Load network and generate random triangle-based HOIs (CM_HOI)
%   2. Build triangle-neighbor mapping (expanded_matrix)
%   3. Run forward/backward simulations with GPU acceleration
%   4. Output synchronization metrics (r_fw, r_bw)
%
% Author: [Your Name]
% Date: [YYYY-MM-DD]
%% =============================================================

%% Step 0: Load the network
data = load('.\real_network\dolphins.mat');
CM = double(data.adj_matrix); 
if issparse(CM)
    CM = full(CM);
end
CM(CM > 0) = 1;
N = size(CM, 1);  % Number of nodes

K = gpuArray(sum(CM, 2)');
K = mean(K, 'all');  % Average degree

%% Step 1: Construct random 3-node hyperedges (triangles)
num_triplets_to_keep = round(K * 100);  % Number of triplets to keep

% Generate all unique triplets
all_triplets = nchoosek(1:N, 3);
idx = randperm(size(all_triplets, 1), min(num_triplets_to_keep, size(all_triplets,1)));
selected_triplets = all_triplets(idx, :);

% Build 3rd-order adjacency tensor CM_HOI
CM_HOI = zeros(N, N, N);
for i = 1:size(selected_triplets, 1)
    a = selected_triplets(i, 1);
    b = selected_triplets(i, 2);
    c = selected_triplets(i, 3);
    % Fill all permutations for symmetry
    CM_HOI(a, b, c) = 1; CM_HOI(a, c, b) = 1;
    CM_HOI(b, a, c) = 1; CM_HOI(b, c, a) = 1;
    CM_HOI(c, a, b) = 1; CM_HOI(c, b, a) = 1;
end

% Count triangle participation per node
triangle_matrix = cell(N, 1);
triangle_count = zeros(N, 1);
for i = 1:N
    [j, k] = find(squeeze(CM_HOI(i, :, :)));
    unique_triplets = unique(sort([i*ones(size(j)), j(:), k(:)], 2), 'rows');
    triangle_matrix{i} = unique_triplets;
    triangle_count(i) = size(unique_triplets, 1);
end

% Build neighbor triangle structure: expanded_matrix
max_triangles = max(triangle_count);
final_matrix = cell(max_triangles, N);
for i = 1:N
    tris = triangle_matrix{i};
    for t = 1:size(tris, 1)
        final_matrix{t, i} = tris(t, :);
    end
end

% Construct 3D triangle-neighbor matrix
max_neighbors = 0;
expanded_matrix = cell(max_triangles, N, max_triangles);
for row = 1:max_triangles
    row
    for col = 1:N
        T = final_matrix{row, col};
        if isempty(T), continue; end
        related_triangles = {};
        for r = 1:max_triangles
            for c = 1:N
                current = final_matrix{r, c};
                if isempty(current) || isequal(sort(current), sort(T))
                    continue;
                end
                if any(ismember(current, T))
                    related_triangles{end+1} = current;
                end
            end
        end
        max_neighbors = max(max_neighbors, length(related_triangles));
        expanded_matrix(row, col, 1:length(related_triangles)) = related_triangles;
    end
end
expanded_matrix = expanded_matrix(:, :, 1:max_neighbors);

% Extract GPU-ready mask/index structure
trangle_matrix_3d = expanded_matrix;
mask = ~cellfun('isempty', trangle_matrix_3d);
all_indices = cell2mat(trangle_matrix_3d(mask));
all_indices = gpuArray(all_indices);
mask = gpuArray(mask);
linear_indices = find(mask);
linear_indices = gpuArray(linear_indices);
avrg = gpuArray(squeeze(sum(sum(mask, 1), 3)));
avrg(avrg == 0) = 1;

clear mask expanded_matrix final_matrix triangle_matrix;

%% Step 2: Simulation parameters
CM = gpuArray(CM);
CM_HOI = gpuArray(CM_HOI);
trangle_matrix_3d0 = gpuArray(zeros(size(trangle_matrix_3d)));
K2 = gpuArray(triangle_count); K2 = mean(K2, 'all');
K = mean(gpuArray(sum(CM, 2)), 'all');

delta = 1;
T_sim = 20;
T0 = 100;
T_tot = T_sim + T0;
step = 0.01;
Time = round(T_tot / step);

w_x = gpuArray(2 * rand(1, N) - 1);  % Natural frequencies
x = gpuArray(pi * (-1 + 2 * rand(1, N)));  % Initial phases
x00 = x;
tt = 100;  % Sampling interval

J1 = -4:0.2:4;  % Coupling range
J2 = 5;         % Fixed higher-order strength
epsilon_list = [0.1 0.3 1 2];

%% Step 3: Simulate for each epsilon
for epsilon = epsilon_list
    fprintf('\nRunning simulation for epsilon = %.2f\n', epsilon);
    x = gpuArray(pi * (-1 + 2 * rand(1, N)));
    r_fw = zeros(size(J1));
    r_bw = zeros(size(J1));
    s_fw = zeros(length(J1), N);
    s_bw = zeros(length(J1), N);
    result_matrix = gpuArray(complex(trangle_matrix_3d0));

    %% Forward scan
    x0 = x;
    for i = 1:length(J1)
        nn = 1; J1(i)
        for t = 1:Time
            e0 = exp(1i * pi * (mod(x0, 2*pi) - pi));
            sums = 1/3 * sum(e0(all_indices), 2);
            result_matrix(linear_indices) = sums;
            adaptive = abs(squeeze(sum(sum(result_matrix, 1), 3)) ./ avrg);
            adaptive_x = epsilon.^(2 * abs(adaptive) - 1);
            adaptive_x(avrg == 1) = 1;
            [F1_x, F2_x] = funx_force(x0, J1(i), adaptive_x .* J2, N, K, K2, CM, CM_HOI);
            x = x0 + step * (w_x + F1_x + F2_x);
            x0 = x;
            if t > T0 / step && mod(t, tt) == 0
                re = mean(cos(x)); im = mean(sin(x));
                r(nn) = sqrt(re^2 + im^2);
                sigma(nn, :) = adaptive_x;
                nn = nn + 1;
            end
        end
        r_fw(i) = mean(r); s_fw(i, :) = mean(sigma, 1);
        if r_fw > 0.7
            K_fw_cross = J1(i);  % Record crossing point
        end
    end

    %% Backward scan
    x0 = x00;
    for i = length(J1):-1:1
        nn = 1; J1(i)
        for t = 1:Time
            e0 = exp(1i * pi * (mod(x0, 2*pi) - pi));
            sums = 1/3 * sum(e0(all_indices), 2);
            result_matrix(linear_indices) = sums;
            adaptive = abs(squeeze(sum(sum(result_matrix, 1), 3)) ./ avrg);
            adaptive_x = epsilon.^(2 * abs(adaptive) - 1);
            adaptive_x(avrg == 1) = 1;
            [F1_x, F2_x] = funx_force(x0, J1(i), adaptive_x .* J2, N, K, K2, CM, CM_HOI);
            x = x0 + step * (w_x + F1_x + F2_x);
            x0 = x;
            if t > T0 / step && mod(t, tt) == 0
                re = mean(cos(x)); im = mean(sin(x));
                r(nn) = sqrt(re^2 + im^2);
                sigma(nn, :) = adaptive_x;
                nn = nn + 1;
            end
        end
        r_bw(i) = mean(r); s_bw(i, :) = mean(sigma, 1);
    end
end

%% Auxiliary function: Compute pairwise and HOI forces
function [F1_x, F2_x] = funx_force(x, J1, J2, N, k, k2, A, A_HOI)
F1_x = (J1 ./ k) .* sum(A .* sin(x' - x), 1);
c1 = 2 * x' - x;
x_hoi = repmat(reshape(x, 1, 1, []), N, N, 1);
cc1 = sin(c1 - x_hoi);
F2 = (J2 ./ (2 .* k2)) .* A_HOI .* cc1;
F2_x = squeeze(sum(sum(F2, 1), 2)); 
F2_x = F2_x(:)';
end
