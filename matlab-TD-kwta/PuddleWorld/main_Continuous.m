% function [weights,data,convergence] = main_Continuous(functionApproximator,nMeshx,nMeshy,nTilex,nTiley,withBias)
% main_continuous.m Performs on-policy sarsa iterative action value funtion 
% estimation for the grid world problem with a puddle. 
% 1) The code can be modified to be continuous state space
% 2) Tile-like modification
%
% Written by:
% Jacob Rafati    1/18/2015
% Jacob Rafati    start of project: 07/01/2014
% email: jrafatiheravi@ucmerced.edu

clc, close all, clear all;
withBias = false;

nMeshx = 20; nMeshy = 20;
nTilex = 1; nTiley = 1;

functionApproximator = 'kwtaNN';
shunt = 1.0;

% control task could be 'grid_world' or 'puddle_world'
task = 'puddle_world';
% function approximator can be either 'kwtaNN' or 'regularBPNN'


% goal in continouos state
s_end = [1.0,1.0];

% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;

% the number of states -- This is the gross mesh states ; the 1st tiling 
nStates = length(xInputInterval) * length(yInputInterval); 

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

%% kwta and regular BP Neural Network
% Weights from input (x,y) to hidden layer
InputSize = length(xInputInterval) + length(yInputInterval);
nCellHidden = round(0.5 * nStates);
mu = 0.1;
Wih = mu * (rand(InputSize,nCellHidden) - 0.5);
biasih = mu * ( rand(1,nCellHidden) - 0.5 );
% biasih = zeros(1,nCellHidden);
% Weights from hidden layer to output
Who = mu * (rand(nCellHidden,nActions) - 0.5);
biasho = mu * ( rand(1,nActions) - 0.5 );
% biasho = zeros(1,nActions);

%% Linear Neural Net
mu = 0.1; % amplitude of random weights
Wio = mu * (rand(InputSize,nActions) - 0.5);
biasio = mu * (rand(1,nActions) - 0.5 );

%% Q Table
Qtable = zeros(nStates,nActions);

%% weights
weights = struct;
Qtable = zeros(nStates,nActions);
weights.Wih = Wih;
weights.Who = Who;
weights.biasih = biasih;
weights.biasho = biasho;
weights.biasio = biasio;
weights.Wio = Wio;
weights.Qtable = Qtable;

data = struct;
meanDeltaForEpisode = [];
varianceDeltaForEpisode = [];
stdDeltaForEpisode = [];
data.meanDeltaForEpisode = meanDeltaForEpisode;
data.varianceDeltaForEpisode = varianceDeltaForEpisode;
data.stdDeltaForEpisode = stdDeltaForEpisode;
initialWinners = [];
ultimateWinners = [];
data.initialWinners = initialWinners;
data.ultimateWinners = ultimateWinners;
%% RUN
if withBias,
    wB = 'withBias';
else
    wB = 'withoutBias';
end

[weights,data,convergence] = sarsa_gridworld(task,functionApproximator,withBias,s_end,weights,data,nMeshx,nTilex,nMeshy,nTiley);