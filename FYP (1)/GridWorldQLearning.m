function GridWorldQLearning(options)
% GridWorldQLearning: implements the simple grid world problem using the
% Q-Learning method

% You can pass the parameters of the problem, through the options
% structure, otherwise a default settings is used for running the program.

% written by: Sina Iravanian - June 2009
% sina@sinairv.com
% Please send your comments or bug reports to the above email address.

if(nargin < 1),     %checks if any options given as argument
    alpha = 0.1;        %learning rate
    gamma = 0.9;        %discount factor
	epsilon = 0.1;      %exploration vs exploitation

	gridcols = 10;      %sets number of columns in grid
	gridrows = 7;        %sets number of rows in grid
	fontsize = 16;      %sets the fontsize
    showTitle = 1;      %sets whether to show title of graph of not 

	episodeCount = 1500;    %sets the number of iterations i.e. episodes
	selectedEpisodes = [episodeCount];   %selects episodes to visualize

	isKing = 0;         %checks if agent can move diagonally like king in chess
	canHold = 0;    % checks if agent can stay in same state/position

    start.row = 7;      %sets starting row position
    start.col = 1;      %sets starting column position
    goal.row = 7;       %sets goal column position
    goal.col = 10;       %sets goal column position
    
    cliff.row = 7;      %set cliff position
    cliff.col =2;
    
    cliff2.row = 7;      %set cliff position
    cliff2.col =3;
    
    cliff3.row = 7;      %set cliff position
    cliff3.col =4;
    
    cliff4.row = 7;      %set cliff position
    cliff4.col =5;
    
    cliff5.row = 7;      %set cliff position
    cliff5.col =6;
    
    cliff6.row = 7;      %set cliff position
    cliff6.col =7;
    
    cliff7.row = 7;      %set cliff position
    cliff7.col =8;
    
    cliff8.row = 7;      %set cliff position
    cliff8.col =9;
    
    cliffArray = [cliff cliff2 cliff3 cliff4 cliff5 cliff6 cliff7 cliff8];
    numCliffs = numel(cliffArray);
    
    reward=0;
    actionsTakenCount=0;
    actionCountArray = zeros(episodeCount,1);
    episodeCountArray = transpose(1:episodeCount);
    episodeRewardArray = zeros(episodeCount,1);
    
    
    actionsTakenCountSarsa=0;
    actionCountArraySarsa = zeros(episodeCount,1);
    episodeRewardArraySarsa = zeros(episodeCount,1);
else  
    
    %set the variables as below if options are given in argument
    gamma = options.gamma;
    alpha = options.alpha;
    epsilon = options.epsilon;

    gridcols = options.gridcols; 
    gridrows = options.gridrows;
    fontsize = options.fontsize;
    showTitle = options.showTitle;

    episodeCount = options.episodeCount;
    selectedEpisodes = options.selectedEpisodes;

    isKing = options.isKing; 
    canHold = options.canHold;

    start = options.start; 
    goal = options.goal;
end

selectedEpIndex = 1;

%if agent can move like king in chess it has 8 actions otherwise 4 NSEW
if(isKing ~= 0),  actionCount = 8; else actionCount = 4; end


%if agent can hold it has an extra action
if(canHold ~= 0 && isKing ~= 0), actionCount = actionCount + 1; end


% initialize Q 3d array with of in this case 7x10x4

%-----------Initialize Q(s, a), ?s ? S, a ? A(s), arbitrarily, 
%and Q(terminal-state, ·) = 0
Q = zeros(gridrows, gridcols, actionCount);


a = 0; % an invalid action



% loop through episodes

%for an array from 1 to the number of eisodes in this case 1000

%-----------Repeat (for each episode):
for ei = 1:episodeCount,

    %disp(sprintf('Running episode %d', ei));
    curpos = start;      %start has property .row and .col
    nextpos = start;
    
    %epsilon or greedy
    
    %takes random number between 0 and 1 and if above epsilon(0.1) 
    %then the max q value action will be taken
    %epsilon = epsilon - ei*(epsilon/episodeCount);
    %disp(epsilon);
    if(rand > epsilon) % greedy
        
        %sets the qmax and action to be take
        [qmax, a] = max(Q(curpos.row,curpos.col,:));
        %if(qmax>-1)
       %     disp([qmax, a]);
       % end
     
    % if below epsilon then the agent takes an exploratory random action
    % NSEW
    else
        a = IntRand(1, actionCount);
        
        
        
        
    end
    
    %while the difference between the current position and goal is not 0
    
    while(PosCmp(curpos, goal) ~= 0)
        
        % take action a, observe r, and nextpos
        %function nextPos = GiveNextPos(curPos, actionIndex, gridCols, gridRows)
        %this sets the next position b incrementing/decrementing the row or
        %column depending on action
        nextpos = GiveNextPos(curpos, a, gridcols, gridrows);
        
        %check if next position is the goal; if so reward= 0 else r=-1
        %observe R, S'
        if(PosCmp(nextpos, goal) ~= 0)
            r = -1; 
            reward=reward+r;
        else
            r = 0;
             
        end
        
        for element = 1:numCliffs,
            %disp('cliff');
            %disp(cliffArray(element));
            if(PosCmp(nextpos, cliffArray(element)) == 0), r = -1000; end
            %if(r==-1000),disp(cliffArray(element)); end
        end
        % choose a_next from nextpos
        
        [qmax, a_next] = max(Q(nextpos.row,nextpos.col,:));
        
        
        
        if(rand <= epsilon) % explore
            a_next = IntRand(1, actionCount);
            
            %disp(a_next);
        end
        
        % update Q: Q-Learning
        curQ = Q(curpos.row, curpos.col, a);  %Q(S,A)
        nextQ = qmax; %Q(nextpos.row, nextpos.col, a_next); Q(S+1,A+1)
        
        %Q(S, A) <- Q(S, A) + a[R + y*maxQ(S', a) - Q(S, A)]
        Q(curpos.row, curpos.col, a) = curQ + alpha*(r + gamma*nextQ - curQ);
        
        %S<-S' 
        curpos = nextpos; a = a_next;
        actionsTakenCount = actionsTakenCount+1;
    end % states in each episode
    
    % if the current state of the world is going to be drawn ...
        
    if(selectedEpIndex <= length(selectedEpisodes) && ei == selectedEpisodes(selectedEpIndex))
        curpos = start;
        rows = []; cols = []; acts = [];
        for i = 1:(gridrows + gridcols) * 10,
            [qmax, a] = max(Q(curpos.row,curpos.col,:));
            nextpos = GiveNextPos(curpos, a, gridcols, gridrows);
            rows = [rows curpos.row];
            %disp('rows=') 
            %disp(curpos.row)
            cols = [cols curpos.col];
            acts = [acts a];

            if(PosCmp(nextpos, goal) == 0), break; end
            curpos = nextpos;
            
        end % states in each episode
        
        %figure;
        figure('Name',sprintf('Episode: %d', ei), 'NumberTitle','off');
        DrawEpisodeState(rows, cols, acts, start.row, start.col, goal.row, goal.col,cliffArray, gridrows, gridcols, fontsize);
        %DrawEpisodeState(rows, cols, acts, start.row, start.col, goal.row, goal.col,cliff.row, cliff.col,cliff2.row, cliff2.col, gridrows, gridcols, fontsize);

        if(showTitle == 1),
            title(sprintf('Simple grid-world Q-Learning - episode %d - (\\epsilon: %3.3f), (\\alpha = %3.4f), (\\gamma = %1.1f)', ei, epsilon, alpha, gamma));
        end
        
        selectedEpIndex = selectedEpIndex + 1;
    end
    episodeRewardArray(ei) = reward/actionsTakenCount;
   actionCountArray(ei)=actionsTakenCount;
   actionsTakenCount=0;
   reward=0;
   
end % episodes loop
%%%%SARSA
Q = zeros(gridrows, gridcols, actionCount);
selectedEpIndex = 1;
epsilon=0.1;
for ei = 1:episodeCount,
    %disp(sprintf('Running episode %d', ei));
    curpos = start;
    nextpos = start;
    
    %epsilon or greedy
    
     %takes random number between 0 and 1 and if above epsilon(0.1) 
    %then the max q value action will be taken
    %epsilon = epsilon - ei*(epsilon/episodeCount);
    %disp(epsilon);
    if(rand > epsilon) % greedy
        %sets the qmax and action to be take
        [qmax, a] = max(Q(curpos.row,curpos.col,:));
      
     
    % if below epsilon then the agent takes an exploratory random action
    % NSEW
    else
        a = IntRand(1, actionCount);
        
        %Pt(a)=exp(qt(a)/T)/ 
        
        Q(nextpos.row, nextpos.col, a_next);
            denominator=0;
            for i = 1:actionCount 
                
                denominator=denominator+ exp(qt(i)/T);
            
            end;
    end

    while(PosCmp(curpos, goal) ~= 0)
        % take action a, observe r, and nextpos
        % take action a, observe r, and nextpos
        %function nextPos = GiveNextPos(curPos, actionIndex, gridCols, gridRows)
        %this sets the next position b incrementing/decrementing the row or
        %column depending on action
        
        nextpos = GiveNextPos(curpos, a, gridcols, gridrows);
        if(PosCmp(nextpos, goal) ~= 0)
            r = -1;
            reward=reward+r;
        else r = 0; 
        end
        
         for element = 1:numCliffs,
            %disp('cliff');
            %disp(cliffArray(element));
            if(PosCmp(nextpos, cliffArray(element)) == 0)
                r = -1000;
                
                
                
                
                
            end
            
            %if(r==-1000),disp(cliffArray(element)); end
        end

        % choose a_next from nextpos
        if(rand > epsilon) % greedy
            [qmax, a_next] = max(Q(nextpos.row,nextpos.col,:));
        else
            %disp('randomJump');
            %disp(rand);
            a_next = IntRand(1, actionCount);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        % update Q: Sarsa
        %Sdisp('test');
        %disp(curpos.row);
        %disp(curpos.col);
        curQ = Q(curpos.row, curpos.col, a);
        nextQ = Q(nextpos.row, nextpos.col, a_next);
        %Q(S, A) <- Q(S, A) + a[R + yQ(S', A') - Q(S, A)]

        Q(curpos.row, curpos.col, a) = curQ + alpha*(r + gamma*nextQ - curQ);
    
        curpos = nextpos; a = a_next;
        actionsTakenCountSarsa = actionsTakenCountSarsa+1;
    end % states in each episode
    %disp('test2');
    % if the current state of the world is going to be drawn ...
    if(selectedEpIndex <= length(selectedEpisodes) && ei == selectedEpisodes(selectedEpIndex))
        curpos = start;
        rows = []; cols = []; acts = [];
        for i = 1:(gridrows + gridcols) * 10,
            [qmax, a] = max(Q(curpos.row,curpos.col,:));
            nextpos = GiveNextPos(curpos, a, gridcols, gridrows);
            rows = [rows curpos.row];
            cols = [cols curpos.col];
            acts = [acts a];

            if(PosCmp(nextpos, goal) == 0), break; end
            curpos = nextpos;
        end % states in each episode
        
        %figure;
        
        figure('Name',sprintf('Episode: %d', ei), 'NumberTitle','off');
        DrawEpisodeState(rows, cols, acts, start.row, start.col, goal.row, goal.col,cliffArray, gridrows, gridcols, fontsize);
        %DrawEpisodeState(rows, cols, acts, start.row, start.col, goal.row, goal.col, gridrows, gridcols, fontsize);
        if(showTitle == 1),
            title(sprintf('Simple grid-world SARSA - episode %d - (\\epsilon: %3.3f), (\\alpha = %3.4f), (\\gamma = %1.1f)', ei, epsilon, alpha, gamma));
        end
        
        selectedEpIndex = selectedEpIndex + 1;
    end
    episodeRewardArraySarsa(ei) = reward/actionsTakenCountSarsa;
    actionCountArraySarsa(ei)=actionsTakenCountSarsa;
   actionsTakenCountSarsa=0;
   reward=0;
end % episodes loop





%disp(size(actionCountArray));
%disp(size(episodeCountArray));
%disp((actionCountArray));
%c

plotEpisodeVsActions(episodeCountArray,actionCountArray,actionCountArraySarsa);

plotEpisodeVsRewards(episodeCountArray,episodeRewardArray,episodeRewardArraySarsa)

end

function plotEpisodeVsActions(episodeCountArray,actionCountArray,actionCountArraySarsa)
            %%%%%% plot episode vs number of actions taken for each episode
            figure
            plot(smooth(episodeCountArray,actionCountArray));
            xlabel('episodeCount')
            ylabel('actionCount')
            hold on
            plot(smooth(episodeCountArray,actionCountArraySarsa,3));
            hold off

            legend('Q-learning','Sarsa')
end

function plotEpisodeVsRewards(episodeCountArray,episodeRewardArray,episodeRewardArraySarsa)
    figure
    plot(smooth(episodeCountArray,episodeRewardArray));
    xlabel('episodeCount')
    ylabel('episodeReward/num actions taken')
    hold on
    plot(smooth(episodeCountArray,episodeRewardArraySarsa));

    hold off

    legend('Q-learning','Sarsa')
end




function c = PosCmp(pos1, pos2)
    c = pos1.row - pos2.row;
    if(c == 0)
        c = c + pos1.col - pos2.col;
    end
end

function nextPos = GiveNextPos(curPos, actionIndex, gridCols, gridRows)
nextPos = curPos;
switch actionIndex
   case 1 % east
       nextPos.col = curPos.col + 1;
   case 2 % south
       nextPos.row = curPos.row + 1;
   case 3 % west
       nextPos.col = curPos.col - 1;
   case 4 % north
       nextPos.row = curPos.row - 1;
   case 5 % northeast 
       nextPos.col = curPos.col + 1;
       nextPos.row = curPos.row - 1;
   case 6 % southeast 
       nextPos.col = curPos.col + 1;
       nextPos.row = curPos.row + 1;
   case 7 % southwest
       nextPos.col = curPos.col - 1;
       nextPos.row = curPos.row + 1;
   case 8 % northwest
       nextPos.col = curPos.col - 1;
       nextPos.row = curPos.row - 1;
   case 9 % hold
       nextPos = curPos;
   otherwise
      disp(sprintf('invalid action index: %d', actionIndex));
end
if(nextPos.col <= 0), nextPos.col = 1; end
if(nextPos.col > gridCols), nextPos.col = gridCols; end
if(nextPos.row <= 0), nextPos.row = 1; end
if(nextPos.row > gridRows), nextPos.row = gridRows; end
end

function n = IntRand(lowerBound, upperBound)
n = floor((upperBound - lowerBound) * rand + lowerBound);
end
