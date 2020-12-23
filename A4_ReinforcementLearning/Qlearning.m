%% Initialization
%  Initialize the world, Q-table, and hyperparameters

world = 1;
state = gwinit(world);
Q = rand(state.ysize, state.xsize, 4);
eta = 0.2;
gamma = 0.9;
eps = 0.9;

%% Training loop
%  Train the agent using the Q-learning algorithm.
for i=1:1500 %for each episode
    state = gwinit(world); %initializing a start state
    
    while state.isterminal == 0 %if current state is not terminal state
        worldPos = state.pos; %store current position of robot 
        [a, oa] = chooseaction( Q, worldPos(1), worldPos(2), [1 2 3 4], [1 1 1 1], eps); %choose an action
        state = gwaction(a); %take action
        r = state.feedback; %observe reward
        s = state.pos; %observe next state
        Q(worldPos(1),worldPos(2),a) = ((1-eta)*Q(worldPos(1),worldPos(2),a))+(eta*(r+gamma*(max(Q(s(1),s(2),:)))));
    end
    
    if rem(i, 50) == 0
        figure(1)
        P = getpolicy(Q);
        gwdraw(i, P)
    end
    
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.

% eps = 0;

for i = 1:10
    state = gwinit(world);

    while state.isterminal == 0 %if current state is not terminal state
        worldPos = state.pos; %store current position of robot 
        a = P(worldPos(1),worldPos(2)); %choose an action
        state = gwaction(a); %take action

        gwdraw([], P)
%         pause(0.01)

        % r = world.feedback; %observe reward
        % s = world.pos; %observe next state
        % Q(worldPos(1),worldPos(2),a) = ((1-eta)*Q(worldPos(1),worldPos(2),a))+(eta*(r+gamma*(max(Q(s(1),s(2),:)))));
    end
end
% figure(2)
% P = getpolicy(Q);
% gwdraw(i, P)