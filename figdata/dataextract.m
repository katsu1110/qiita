%% generate artificial data and plot 
close all;
figure;
xdata = rand(1,30);
ydata = rand(1,30);
scatter(xdata,ydata,30,'filled')

%% plot a linear regression line
b = glmfit(xdata,ydata,'normal','link','identity');
hold on;
plot([0 1],b(1) + b(2)*[0 1])

%% plot a unity line
hold on;
plot([0 1],[0 1])

%% extract data from the figure
% current figure handle;
h = gcf;

% axis object
axesObjs = get(h, 'Children'); 

% data object inside the axis
dataObjs = get(axesObjs, 'Children'); 

% extract 'scatter' objects
scatters = findobj(dataObjs, 'type', 'scatter');

% extract 'line' objects
lines = findobj(dataObjs, 'type', 'line');

%% play with the extracted objects (scatter)
[xdata,idx] = sort(scatters.XData);
ydata = scatters.YData(idx);
figure;

% - change transparency in a gradient manner with x-axis
% - change color to red
% - change markersize to 100
% - change marker if datapoints are under the unity line
for i = 1:30
    if xdata(i) < ydata(i)
        scatter(xdata(i),ydata(i),100,'filled',...
            'markerfacecolor','r','markeredgecolor','r',...
            'markerfacealpha',i/30,'markeredgealpha',i/30);
    else
        scatter(xdata(i),ydata(i),100,'filled','s',...
            'markerfacecolor','r','markeredgecolor','r',...
            'markerfacealpha',i/30,'markeredgealpha',i/30);
    end
    hold on;
end

%% play with the extracted objects (line)
% - change width of the both lines
% - green dash regression line
% - black dotted unity line

% regression line
plot(lines(2).XData, lines(2).YData, '--g','linewidth', 2)
hold on;

% unity line
plot(lines(1).XData, lines(1).YData, ':k','linewidth', 2)

%% other cosmetics
% box off
set(gca, 'box', 'off')

% tick out
set(gca, 'TickDir', 'out')

% remove some ticks
set(gca, 'XTick',[0 0.5 1])
set(gca, 'YTick',[0 0.5 1])

% larger font size
set(gca,'FontSize',18)

% square axis
axis square
