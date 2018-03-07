nMeshp = 60;
nMeshv = 60;
grafica = true;
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');
set(0,'DefaultAxesFontSize',18);
set(0,'DefaultTextFontSize',18);
set(0, 'defaultTextInterpreter', 'latex');


for i=1:4,
   if i==1, functionApproximator='kwtaNN';
   elseif i==2, functionApproximator='linearNN';
   elseif i==3, functionApproximator='regularBPNN';
   elseif i==4, functionApproximator='Qtable';
   end
   filename = strcat('results/',functionApproximator,'-nMeshp-',num2str(nMeshp),'-nMeshv-',num2str(nMeshv),'.mat');  
   load(filename);
   
   h = figure;
   
        Q{i} = plotQ(weights,functionApproximator,nMeshp,nMeshv,grafica);        
        axis tight
        view(30,45)
        title(['Value function using ',functionApproximator])
        xlabel('Position')
        ylabel('Velocity')

        ax = get(gca,'XLabel'); % Handle of the x label
        set(ax, 'Units', 'Normalized')
        pos = get(ax, 'Position');
        axx = gca;
        axx.XTickLabel = {-1.5,'','','',0.6};
        axx.YTickLabel = {-0.07,'',0.07};
        set(ax, 'Position',pos.*[1,1,1],'Rotation',-20)
        ay = get(gca,'YLabel'); % Handle of the y label
        set(ay, 'Units', 'Normalized')
        pos = get(ay, 'Position');
        set(ay, 'Position',pos.*[1,1,1],'Rotation',45)
        
%         fig = gcf;
%         fig.PaperUnits = 'inches';
%         fig.PaperPosition = [0 0 10 10];
%         fig.PaperPositionMode = 'manual';
        
        figname = ['figures/',functionApproximator,'-cost'];        
        print(h,figname,'-dpdf')
        print(h,figname,'-depsc')
                
   h = figure;   
        subplot(2,1,1);
        plot(data.meanDeltaForEpisode)
        
        axx = gca;
        axx.XTickLabel = {0,50000,100000,150000,200000};
        ylabel('mean(\delta) in episode')
        title(['Performance for ',functionApproximator])
        subplot(2,1,2); 
        plot(data.steps)
        xlabel('Episode number')
        ylabel('number of steps')
        axx = gca;
        axx.XTickLabel = {0,50000,100000,150000,200000};
        figname = ['figures/',functionApproximator,'-performance'];        
        print(h,figname,'-dpdf')
        print(h,figname,'-depsc')

        
    h=figure;
    
        subplot(2,1,1);
        xlabel('Episode number')
        plot(data.meanDeltaForEpisodeTest)
        axx = gca;
        axx.XTickLabel = {0,50000,100000,150000,200000};
        ylabel('mean(\delta) in episode')
        title(['Test with frozen weights for ',functionApproximator])    
        %% 
        subplot(2,1,2); 
        plot(data.stepsTest)
        xlabel('Episode number')
        ylabel('number of steps')
        axx = gca;
        axx.XTickLabel = {0,50000,100000,150000,200000};
        figname = ['figures/',functionApproximator,'-frozen-test'];        
        print(h,figname,'-dpdf')
        print(h,figname,'-depsc')

end
