clear; clc; close all;
rng(77);  % Set random seed

%% ============================================================
% File: simulate_hoi_dynamics_Japanese_macaques.m
% Description:
%   This script simulates synchronization dynamics on a real-world
%   network (Japanese macaques) using HpOI (HOIs).
%   Triangle-based HOIs are extracted directly from the adjacency matrix.
%   The model supports forward and backward bifurcation scanning across J1
%   for various adaptive exponent values (epsilon).
%
% Outputs: forward/backward order parameters (r_fw, r_bw)

%% ============================================================

%% Step 1: Load adjacency matrix
data = load('.\real_network\Japanese_macaques.mat');
CM = double(data.adj_matrix);
if issparse(CM), CM = full(CM); end
CM(CM > 0) = 1;  % Binarize
N = size(CM, 1); % Number of nodes

%% Step 2: Detect triangles (i,j,k such that i-j-k-i forms a loop)
triangles = cell(N, 1);
parfor i = 1:N
    localTriangles = [];
    j_nodes = find(CM(i, :) > 0);
    for j = j_nodes
        k_nodes = find(CM(j, :) > 0);
        for k = k_nodes
            if CM(k, i) > 0
                localTriangles = [localTriangles; i, j, k];
            end
        end
    end
    triangles{i} = localTriangles;
end
triangles = round(vertcat(triangles{:}));

%% Step 3: Build HOI tensor CM_HOI (3rd-order)
CM_HOI = zeros(N, N, N);
for i = 1:size(triangles, 1)
    a = triangles(i, 1); b = triangles(i, 2); c = triangles(i, 3);
    CM_HOI(a,b,c) = 1; CM_HOI(a,c,b) = 1;
    CM_HOI(b,a,c) = 1; CM_HOI(b,c,a) = 1;
    CM_HOI(c,a,b) = 1; CM_HOI(c,b,a) = 1;
end

%% Step 4: Build triangle matrix and expansion structure
triangle_matrix = cell(N, 1);
triangle_count = zeros(1, N);
for i = 1:N
    tris = [];
    for j = 1:N
        for k = j+1:N
            if CM_HOI(i,j,k) == 1
                tris = [tris; i, j, k];
            end
        end
    end
    triangle_matrix{i} = tris;
    triangle_count(i) = size(tris, 1);
end

max_triangles = max(cellfun(@(x) size(x,1), triangle_matrix));
final_matrix = cell(max_triangles, N);
for i = 1:N
    T = triangle_matrix{i};
    if isempty(T), continue; end
    final_matrix(1:size(T,1), i) = mat2cell(T, ones(size(T,1), 1), 3);
end

% Expand neighboring triangle structure
expanded_matrix = cell(max_triangles, N, max_triangles);
max_neighbors = 0;
for row = 1:max_triangles
    row
    for col = 1:N
        T = final_matrix{row, col};
        if isempty(T), continue; end
        related_triangles = {};
        for r = 1:max_triangles
            for c = 1:N
                current = final_matrix{r, c};
                if isempty(current), continue; end
                if any(ismember(current, T)) && ~isequal(sort(current), sort(T))
                    related_triangles{end+1} = current;
                end
            end
        end
        max_neighbors = max(max_neighbors, length(related_triangles));
        expanded_matrix(row, col, 1:length(related_triangles)) = related_triangles;
    end
end
expanded_matrix = expanded_matrix(:, :, 1:max_neighbors);

%% Step 5: Preprocessing: GPU data & index setup
CM = gpuArray(CM);
CM_HOI = gpuArray(CM_HOI);
trangle_matrix_3d = expanded_matrix;
trangle_matrix_3d0 = gpuArray(zeros(size(trangle_matrix_3d)));

mask = ~cellfun('isempty', trangle_matrix_3d);
all_indices = gpuArray(cell2mat(trangle_matrix_3d(mask)));
linear_indices = gpuArray(find(mask));
avrg = gpuArray(squeeze(sum(sum(mask, 1), 3)));
avrg(avrg == 0) = 1;

clear mask trangle_matrix_3d expanded_matrix final_matrix related_triangles;

%% Step 6: Simulation parameters
N_nod = N;
K = mean(gpuArray(sum(CM, 2)), 'all');
K2 = mean(gpuArray(triangle_count), 'all');

delta = 1;
T_sim = 20; T0 = 100; T_tot = T_sim + T0;
step = 0.01;
Time = round(T_tot / step);
w_x = gpuArray(2 * rand(1, N) - 1); 
x = gpuArray(pi * (-1 + 2 * rand(1, N))); 
x00 = x;
tt = 100;
J1 = -4:0.2:4; J2 = 8;

epsilon_list = [2 3 4 5 6 7 8];  % Adaptive strength

%% Step 7: Simulation over different epsilon
for epsilon = epsilon_list
    fprintf('\nRunning simulation for epsilon = %.2f\n', epsilon);

    x = gpuArray(pi*(-1 + 2*rand(1, N_nod)));
    r_fw = zeros(size(J1)); r_bw = zeros(size(J1));
    s_fw = zeros(length(J1), N_nod);
    s_bw = zeros(length(J1), N_nod);
    result_matrix = gpuArray(complex(trangle_matrix_3d0));

    % Forward scan
    for i = 1:length(J1)
        nn = 1; x0 = x; J1(i)
        for t = 1:Time
            e0 = exp(1i * pi * (mod(x0, 2*pi) - pi));
            sums = 1/3 * sum(e0(all_indices), 2);
            result_matrix(linear_indices) = sums;
            adaptive = abs(squeeze(sum(sum(result_matrix, 1), 3)) ./ avrg);
            adaptive_x = epsilon.^(2 * adaptive - 1);
            adaptive_x(avrg == 1) = 1;
            [F1_x, F2_x] = funx_force(x0, J1(i), adaptive_x .* J2, N_nod, K, K2, CM, CM_HOI);
            x = x0 + step * (w_x + F1_x + F2_x); x0 = x;

            if t > T0 / step && mod(t, tt) == 0
                re = mean(cos(x)); im = mean(sin(x));
                r(nn) = sqrt(re^2 + im^2);
                sigma(nn, :) = adaptive_x;
                nn = nn + 1;
            end
        end
        r_fw(i) = mean(r); s_fw(i,:) = mean(sigma,1);
    end

    % Backward scan
    x0 = x00;
    for i = length(J1):-1:1
        nn = 1; J1(i)
        for t = 1:Time
            e0 = exp(1i * pi * (mod(x0, 2*pi) - pi));
            sums = 1/3 * sum(e0(all_indices), 2);
            result_matrix(linear_indices) = sums;
            adaptive = abs(squeeze(sum(sum(result_matrix, 1), 3)) ./ avrg);
            adaptive_x = epsilon.^(2 * adaptive - 1);
            adaptive_x(avrg == 1) = 1;
            [F1_x, F2_x] = funx_force(x0, J1(i), adaptive_x .* J2, N_nod, K, K2, CM, CM_HOI);
            x = x0 + step * (w_x + F1_x + F2_x); x0 = x;

            if t > T0 / step && mod(t, tt) == 0
                re = mean(cos(x)); im = mean(sin(x));
                r(nn) = sqrt(re^2 + im^2);
                sigma(nn, :) = adaptive_x;
                nn = nn + 1;
            end
        end
        r_bw(i) = mean(r); s_bw(i,:) = mean(sigma,1);
    end
end

%% Auxiliary Function: Compute pairwise + HOI forces
function [F1_x,F2_x]=funx_force(x,J1,J2,N_nod,k,k2,A,A_HOI)
F1_x = (J1 ./ k) .* sum(A .* sin(x' - x), 1);
c1 = 2 * x' - x;
x_hoi = repmat(reshape(x, 1, 1, []), N_nod, N_nod, 1);
cc1 = sin(c1 - x_hoi);
F2 = (J2 ./ (2 .* k2)) .* A_HOI .* cc1;
F2_x = squeeze(sum(sum(F2, 1), 2)); F2_x = F2_x(:)';
end

%% Optional: Lorentzian distribution generator (not used here)
function [w] = Lorentzian_distribution(w0, num_samples)
pdf = @(w) 1 ./ (pi * (1 + (w - w0).^2));
w_values = linspace(-200, 200, num_samples);
cdf_values = cumtrapz(w_values, pdf(w_values));
cdf_values = cdf_values / cdf_values(end);
inverse_cdf = griddedInterpolant(cdf_values, w_values, 'linear');
w = inverse_cdf(rand(num_samples, 1))';
end
